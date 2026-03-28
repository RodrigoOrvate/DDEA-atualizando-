import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import requests
import gzip
import io
import re
import gc
from Bio import Entrez
import statsmodels.api as sm

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleBio/1.0'}
Entrez.email = "rodrigo.arruda@ufrn.edu.br"


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def quantile_normalize(df_values):
    if df_values.size == 0:
        return df_values
    mat = df_values.astype(np.float32)
    sorted_mat = np.sort(mat, axis=0)
    rank_mean = sorted_mat.mean(axis=1).astype(np.float32)
    del sorted_mat; gc.collect()
    indices = np.argsort(mat, axis=0)
    norm_mat = np.empty_like(mat, dtype=np.float32)
    for i in range(mat.shape[1]):
        norm_mat[indices[:, i], i] = rank_mean
    del mat, indices; gc.collect()
    return norm_mat


# ============================================================
# DETECÇÃO DO TIPO DE ID NO ÍNDICE
# ============================================================

def detect_index_type(index_values):
    """
    Analisa os primeiros valores do índice e retorna:
    'ensembl'    → ENSG00000...
    'entrez'     → apenas dígitos
    'symbol'     → já é Gene Symbol (letras+dígitos curtos, tipo TP53, BRCA1)
    'probe'      → probe ID de microarray (ex: 1007_s_at, 11715100_at)
    'unknown'    → não identificado
    """
    samples = [str(v).strip() for v in index_values[:50] if str(v).strip()]
    if not samples:
        return 'unknown'

    ensembl  = sum(1 for s in samples if re.match(r'^ENS[A-Z]*G\d{11}', s))
    entrez   = sum(1 for s in samples if re.match(r'^\d+$', s))
    probe    = sum(1 for s in samples if re.match(r'^\d+_[a-z]', s) or re.match(r'^[A-Z]{1,3}\d{6,}', s))
    # Symbol: string curta sem underscores numéricos, geralmente 2-10 chars, letras+números
    symbol   = sum(1 for s in samples if re.match(r'^[A-Za-z][A-Za-z0-9\-\.]{1,15}$', s)
                   and not re.match(r'^\d+$', s)
                   and '_' not in s)

    scores = {'ensembl': ensembl, 'entrez': entrez, 'probe': probe, 'symbol': symbol}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return 'unknown'
    return best


# ============================================================
# MAPEAMENTO — MICROARRAY (GPL soft file)
# ============================================================

@st.cache_data(show_spinner=False)
def get_gene_mapping_microarray(gse_id):
    try:
        search_handle = Entrez.esearch(db="gds", term=f"{gse_id}[ACCN]")
        uid = Entrez.read(search_handle)["IdList"][0]
        record = Entrez.read(Entrez.esummary(db="gds", id=uid))
        gpl_id = record[0]['GPL']
        gpl_prefix = f"GPL{gpl_id[:-3]}nnn" if len(gpl_id) > 3 else "GPLnnn"
        url = (
            f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl_prefix}/"
            f"GPL{gpl_id}/soft/GPL{gpl_id}_family.soft.gz"
        )
        response = requests.get(url, stream=True, headers=HEADERS)
        with gzip.open(response.raw, 'rt', encoding='utf-8', errors='ignore') as f:
            table_lines, in_table = [], False
            for line in f:
                if line.startswith('!platform_table_begin'):
                    in_table = True; continue
                if line.startswith('!platform_table_end'):
                    break
                if in_table:
                    table_lines.append(line)
                if len(table_lines) > 250000:
                    break
            map_df = pd.read_csv(io.StringIO("".join(table_lines)), sep='\t', low_memory=False)
            target_cols = ['Gene.Symbol', 'Gene Symbol', 'GENE_SYMBOL', 'Symbol', 'SYMBOL']
            symbol_col = next(
                (c for c in map_df.columns if any(k.upper() == c.upper() for k in target_cols)),
                None
            )
            if symbol_col:
                map_df['ID'] = map_df['ID'].astype(str).str.strip().str.replace('"', '')
                map_df[symbol_col] = map_df[symbol_col].astype(str).apply(
                    lambda x: x.split(' /// ')[0]
                )
                res = map_df[['ID', symbol_col]].rename(
                    columns={'ID': 'Probe_ID', symbol_col: 'Symbol'}
                )
                return res, map_df.head(100)
        return None, None
    except Exception as e:
        st.warning(f"Erro no mapeamento Microarray: {e}")
        return None, None


# ============================================================
# MAPEAMENTO — RNA-SEQ
# ============================================================

@st.cache_data(show_spinner=False)
def get_gene_mapping_rnaseq(index_ids: tuple, id_type: str):
    """
    Converte IDs para Gene Symbol conforme o tipo detectado.
    - entrez  → MyGene.info POST com field=symbol
    - ensembl → MyGene.info POST com field=symbol, scopes=ensembl.gene
    - symbol  → já é símbolo, retorna identidade
    - probe/unknown → retorna None (não tenta)
    """
    if id_type == 'symbol':
        # Já são símbolos — mapeamento identidade
        df = pd.DataFrame({'Probe_ID': list(index_ids), 'Symbol': list(index_ids)})
        return df, "Índice já contém Gene Symbols — nenhuma conversão necessária."

    if id_type not in ('entrez', 'ensembl'):
        return None, f"Tipo de ID '{id_type}' não suportado para conversão automática."

    if id_type == 'entrez':
        ids_clean = [str(i).strip() for i in index_ids if str(i).strip().isdigit()]
        scope = "entrezgene"
    else:
        ids_clean = [str(i).strip() for i in index_ids if str(i).strip().startswith('ENS')]
        scope = "ensembl.gene"

    if not ids_clean:
        return None, "Nenhum ID válido encontrado para conversão."

    results = []
    for i in range(0, len(ids_clean), 1000):
        chunk = ids_clean[i:i + 1000]
        try:
            resp = requests.post(
                "https://mygene.info/v3/query",
                data={
                    "q": ",".join(chunk),
                    "scopes": scope,
                    "fields": "symbol",
                    "species": "human",
                },
                timeout=30,
            )
            resp.raise_for_status()
            for item in resp.json():
                if item.get('notfound'):
                    continue
                query_id = str(item.get('query', ''))
                symbol = item.get('symbol', None)
                results.append({"Probe_ID": query_id, "Symbol": symbol})
        except Exception as e:
            return None, f"MyGene.info erro: {e}"

    if not results:
        return None, "MyGene.info não retornou resultados."

    mapping_df = pd.DataFrame(results).drop_duplicates('Probe_ID')
    mapping_df["Symbol"] = mapping_df["Symbol"].fillna(mapping_df["Probe_ID"])
    n_mapped = (mapping_df['Symbol'] != mapping_df['Probe_ID']).sum()
    return mapping_df, f"{n_mapped}/{len(mapping_df)} IDs convertidos para Gene Symbol via MyGene.info ({scope})."


# ============================================================
# LEITURA E PARSE DE BYTES
# ============================================================

def _parse_matrix_bytes(raw_bytes):
    """
    Tenta ler bytes (gz ou não) como tabela de expressão.
    Retorna DataFrame com índice=genes, colunas=amostras, ou None.
    """
    try:
        try:
            content = gzip.decompress(raw_bytes)
        except Exception:
            content = raw_bytes

        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, index_col=0, low_memory=False)
                df.index = df.index.astype(str).str.strip().str.replace('"', '')
                df.columns = [str(c).strip().replace('"', '') for c in df.columns]
                num_cols = df.select_dtypes(include=[np.number]).shape[1]
                if num_cols >= 2 and df.shape[0] >= 10:
                    # Mantém apenas colunas numéricas
                    df = df.select_dtypes(include=[np.number])
                    return df
            except Exception:
                continue
        return None
    except Exception:
        return None


# ============================================================
# SERIES MATRIX — extração de metadados + matriz
# ============================================================

def _try_series_matrix(gse_id):
    """
    Baixa a series_matrix.
    Retorna (df_expression_or_None, meta_df, gsms, gsm_order).
    gsm_order é a lista ordenada de GSMs conforme o cabeçalho.
    """
    num = gse_id.replace("GSE", "")
    prefix = f"GSE{num[:-3]}nnn" if len(num) > 3 else "GSEnnn"
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}"
        f"/matrix/{gse_id}_series_matrix.txt.gz"
    )
    try:
        r = requests.get(url, timeout=60, headers=HEADERS)
        r.raise_for_status()
        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            titles, gsms, char_lines, gsm_order = [], [], [], []
            df = pd.DataFrame()
            for line in f:
                if line.startswith('!Sample_title'):
                    titles = [t.strip().replace('"', '') for t in line.split('\t')[1:]]
                if line.startswith('!Sample_geo_accession'):
                    gsms = [t.strip().replace('"', '') for t in line.split('\t')[1:]]
                    gsm_order = gsms[:]
                if line.startswith('!Sample_characteristics_ch1'):
                    char_lines.append(line.split('\t')[1:])
                if line.startswith('ID_REF') or line.startswith('"ID_REF"'):
                    df = pd.read_csv(f, sep='\t', header=None, low_memory=True)
                    break

            meta_dict = {"Accession": gsms, "Title": titles}
            for row in char_lines:
                if row:
                    key = row[0].split(':')[0].strip() if ':' in row[0] else f"Info_{len(meta_dict)}"
                    meta_dict[key] = [
                        v.split(': ')[1].strip() if ': ' in v else v.strip()
                        for v in row
                    ][:len(gsms)]
            meta_df = pd.DataFrame(meta_dict)

            if df.empty or len(df.columns) < 2:
                return None, meta_df, gsms, gsm_order

            df = df.set_index(0)
            df.index = df.index.astype(str).str.strip().str.replace('"', '')
            col_rename = {col: gsm_order[i] for i, col in enumerate(df.columns) if i < len(gsm_order)}
            df.rename(columns=col_rename, inplace=True)

            num_cols = df.select_dtypes(include=[np.number]).shape[1]
            if num_cols < 2:
                return None, meta_df, gsms, gsm_order

            df = df.select_dtypes(include=[np.number])
            return df, meta_df, gsms, gsm_order

    except Exception as e:
        return None, None, None, []


def _list_supplementary_urls(gse_id):
    num = gse_id.replace("GSE", "")
    prefix = f"GSE{num[:-3]}nnn" if len(num) > 3 else "GSEnnn"
    base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/suppl/"
    try:
        r = requests.get(base_url, timeout=20, headers=HEADERS)
        if r.status_code != 200:
            return []
        files = re.findall(r'href="([^"]+\.(txt|tsv|csv)(\.gz)?)"', r.text, re.IGNORECASE)
        return [base_url + f[0] for f in files]
    except Exception:
        return []


def _score_suppl_file(url):
    name = url.lower()
    score = 0
    if any(k in name for k in ['count', 'raw', 'read', 'htseq', 'featurecount', 'matrix']):
        score += 3
    if any(k in name for k in ['norm', 'fpkm', 'rpkm', 'tpm', 'cpm']):
        score += 1
    if 'log' in name:
        score -= 1
    return score


def _try_supplementary(gse_id, log_cb=None):
    urls = sorted(_list_supplementary_urls(gse_id), key=_score_suppl_file, reverse=True)
    for url in urls:
        try:
            if log_cb:
                log_cb(f"Suplementar: `{url.split('/')[-1]}`")
            r = requests.get(url, timeout=60, headers=HEADERS)
            r.raise_for_status()
            df = _parse_matrix_bytes(r.content)
            if df is not None and df.shape[1] >= 2 and df.shape[0] >= 10:
                return df
        except Exception:
            continue
    return None


def _try_ncbi_generated(gse_id, log_cb=None):
    candidates = [
        f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_id}&format=file&file={gse_id}_raw_counts_GEO.txt.gz",
        f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_id}&format=file&file={gse_id}_norm_counts_GEO.txt.gz",
    ]
    for url in candidates:
        try:
            if log_cb:
                log_cb(f"NCBI-generated: `{url.split('file=')[-1]}`")
            r = requests.get(url, timeout=60, headers=HEADERS, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1000:
                df = _parse_matrix_bytes(r.content)
                if df is not None and df.shape[1] >= 2 and df.shape[0] >= 10:
                    return df
        except Exception:
            continue
    return None


# ============================================================
# SINCRONIZAÇÃO DE COLUNAS DE SUPLEMENTAR COM GSMs
# ============================================================

def _sync_suppl_columns_with_gsms(df_suppl, gsm_order):
    """
    Tenta alinhar as colunas do arquivo suplementar com os GSMs da series_matrix.

    Estratégias (em ordem):
    1. Interseção direta — colunas já são GSMxxxxxx
    2. As colunas estão na mesma ordem que os GSMs (renomeia posicionalmente)
       — só aplica se o número de colunas bater com o número de GSMs
    3. Falha — retorna df original com aviso

    Retorna (df_sincronizado, método_usado).
    """
    cols = list(df_suppl.columns)
    gsm_set = set(gsm_order)

    # Estratégia 1: interseção direta
    direct = [c for c in cols if c in gsm_set]
    if len(direct) >= 2:
        return df_suppl[direct], "interseção direta (colunas já são GSMxxxxxx)"

    # Estratégia 2: renomeação posicional
    if len(cols) == len(gsm_order):
        rename_map = dict(zip(cols, gsm_order))
        return df_suppl.rename(columns=rename_map), "renomeação posicional (mesma ordem da series_matrix)"

    # Estratégia 3: subset posicional (mais colunas no suppl do que GSMs — ex: coluna de gene_name extra)
    # Tenta ignorar a primeira coluna se já foi usada como índice
    if len(cols) - 1 == len(gsm_order):
        # Provavelmente a primeira coluna é anotação (gene name etc), já deveria ser índice
        df_try = df_suppl.iloc[:, 1:]
        rename_map = dict(zip(df_try.columns, gsm_order))
        return df_try.rename(columns=rename_map), "renomeação posicional ignorando primeira coluna de anotação"

    return df_suppl, "não sincronizado — colunas podem não corresponder aos GSMs"


# ============================================================
# EXTRAÇÃO PRINCIPAL — CASCATA
# ============================================================

def get_geo_full_data(gse_id, mode, log_cb=None):
    """
    Retorna (df_matrix, meta_df, gsms, gsm_order, source, error).
    """
    if log_cb:
        log_cb("Buscando metadados e Series Matrix...")

    df_matrix, meta_df, gsms, gsm_order = _try_series_matrix(gse_id)

    if meta_df is None:
        return None, None, None, [], None, "Falha ao acessar o GEO. Verifique o GSE ID."

    if mode == "Microarray":
        if df_matrix is not None:
            return df_matrix, meta_df, gsms, gsm_order, "Series Matrix", None
        return None, meta_df, gsms, gsm_order, None, None

    # RNA-Seq: cascata
    if df_matrix is not None and df_matrix.shape[1] >= 2:
        return df_matrix, meta_df, gsms, gsm_order, "Series Matrix", None

    if log_cb:
        log_cb("Series Matrix sem dados de expressão. Buscando arquivos suplementares...")
    df_suppl = _try_supplementary(gse_id, log_cb=log_cb)
    if df_suppl is not None:
        df_synced, sync_method = _sync_suppl_columns_with_gsms(df_suppl, gsm_order)
        if log_cb:
            log_cb(f"Sincronização de colunas: {sync_method}")
        return df_synced, meta_df, gsms, gsm_order, f"Supplementary TXT ({sync_method})", None

    if log_cb:
        log_cb("Suplementares não encontrados. Tentando NCBI-generated counts...")
    df_ncbi = _try_ncbi_generated(gse_id, log_cb=log_cb)
    if df_ncbi is not None:
        df_synced, sync_method = _sync_suppl_columns_with_gsms(df_ncbi, gsm_order)
        return df_synced, meta_df, gsms, gsm_order, f"NCBI-generated ({sync_method})", None

    if log_cb:
        log_cb("Nenhuma fonte automática encontrou dados. Upload manual necessário.")
    return None, meta_df, gsms, gsm_order, None, None


# ============================================================
# HELPERS DE ESTADO
# ============================================================

def reset_analysis_state():
    for k in ['df', 'meta_df', 'res', 'analysis_done', 'mapping', 'mapping_msg',
              'raw_gpl', 'mode', 'norm_df', 'rn', 'tn', 'gse_id',
              'gsms', 'gsm_order', 'matrix_source', 'id_type']:
        st.session_state.pop(k, None)
    st.session_state['groups'] = {}
    st.session_state['group_field_key'] = st.session_state.get('group_field_key', 0) + 1


def all_assigned_samples(groups: dict, exclude: str = None) -> set:
    assigned = set()
    for name, samples in groups.items():
        if name != exclude:
            assigned.update(samples)
    return assigned


# ============================================================
# APP PRINCIPAL
# ============================================================

def run_app():
    st.set_page_config(layout="wide", page_title="DDEA Final Master")
    st.title("Diagonal Differential Expression Alley 🧬")

    if 'groups' not in st.session_state:
        st.session_state['groups'] = {}
    if 'group_field_key' not in st.session_state:
        st.session_state['group_field_key'] = 0

    # ----------------------------------------------------------
    # SIDEBAR
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("1. GEO Input")
        mode = st.radio("Experiment Type:", ["Microarray", "RNASeq"])
        gse_input = st.text_input("GSE ID:", value="GSE117769")
        fetch_btn = st.button("🚀 Fetch Data", use_container_width=True)

        if 'meta_df' in st.session_state:
            st.divider()
            if st.session_state.get('matrix_source'):
                st.caption(f"📦 Fonte: **{st.session_state['matrix_source']}**")
            if st.session_state.get('id_type'):
                st.caption(f"🔑 Tipo de ID: **{st.session_state['id_type']}**")
            if st.session_state.get('mapping_msg'):
                st.caption(f"🔗 {st.session_state['mapping_msg']}")

            st.divider()
            st.header("2. Labels Display")
            all_cols = list(st.session_state['meta_df'].columns)
            # Remove Accession das opções extras — será sempre incluído
            extra_cols = [c for c in all_cols if c != 'Accession']
            extra_selected = st.multiselect(
                "Informações adicionais no seletor:",
                extra_cols,
                default=["Title"] if "Title" in extra_cols else extra_cols[:1],
            )
            # Accession sempre forçado como primeiro elemento
            label_cols = ['Accession'] + extra_selected

            st.divider()
            st.header("3. Parameters")
            gene_area = st.text_area("Genes to Highlight (1 per line):")
            p_thr = st.slider("P-value threshold:", 0.001, 0.10, 0.05, format="%.3f")
            fc_thr = st.slider("Min Abs Log2FC:", 0.0, 10.0, 0.0, step=0.1)
            if mode == "Microarray":
                use_limma = st.checkbox("Usar modelo linear (Limma-like)", value=True)
            else:
                use_limma = False
            max_plot = st.number_input("Max genes in plots:", value=50, min_value=5)
        else:
            label_cols = ['Accession', 'Title']
            gene_area = ""
            p_thr, fc_thr, use_limma, max_plot = 0.05, 0.0, False, 20

    # ----------------------------------------------------------
    # FETCH
    # ----------------------------------------------------------
    if fetch_btn and gse_input:
        prev_gse = st.session_state.get('gse_id', '')
        prev_mode = st.session_state.get('mode', '')
        if gse_input.strip() != prev_gse or mode != prev_mode:
            reset_analysis_state()

        log_placeholder = st.empty()
        log_lines = []

        def log_cb(msg):
            log_lines.append(f"› {msg}")
            log_placeholder.info("\n\n".join(log_lines))

        with st.spinner("🚀 Buscando dados do GEO..."):
            df, meta_df, gsms, gsm_order, source, err = get_geo_full_data(
                gse_input.strip(), mode, log_cb=log_cb
            )

        log_placeholder.empty()

        if err:
            st.error(err)
        elif meta_df is None:
            st.error("Falha ao acessar o GEO. Verifique o GSE ID e sua conexão.")
        else:
            st.session_state['mode'] = mode
            st.session_state['gse_id'] = gse_input.strip()
            st.session_state['matrix_source'] = source
            st.session_state['gsm_order'] = gsm_order

            if mode == "Microarray":
                with st.spinner("🔬 Buscando mapeamento GPL..."):
                    map_res, raw_gpl = get_gene_mapping_microarray(gse_input.strip())
                st.session_state['raw_gpl'] = raw_gpl
                st.session_state['mapping'] = map_res
                st.session_state['mapping_msg'] = "Mapeamento via GPL soft file."
                st.session_state['id_type'] = 'probe'
            else:
                # Detecta tipo de ID se tiver matriz
                if df is not None and not df.empty:
                    id_type = detect_index_type(df.index.tolist())
                    st.session_state['id_type'] = id_type
                    with st.spinner(f"🔗 Mapeando IDs ({id_type}) → Gene Symbol..."):
                        mapping, msg = get_gene_mapping_rnaseq(
                            tuple(df.index.astype(str).tolist()), id_type
                        )
                    st.session_state['mapping'] = mapping
                    st.session_state['mapping_msg'] = msg
                else:
                    st.session_state['mapping'] = None
                    st.session_state['mapping_msg'] = ""
                    st.session_state['id_type'] = 'unknown'
                st.session_state['raw_gpl'] = None

            st.session_state.update({
                'df': df,
                'meta_df': meta_df,
                'gsms': gsms,
                'analysis_done': False,
            })

            if source and df is not None:
                st.success(
                    f"✅ Matriz carregada via **{source}** — "
                    f"{df.shape[0]} genes × {df.shape[1]} amostras"
                )
            elif not source:
                st.warning("⚠️ Nenhuma fonte automática encontrou dados. Faça upload manual abaixo.")

            if st.session_state.get('mapping_msg'):
                msg = st.session_state['mapping_msg']
                if "não suportado" in msg or "falhou" in msg.lower() or "Nenhum" in msg:
                    st.warning(f"⚠️ {msg}")
                else:
                    st.info(f"🔗 {msg}")

            st.rerun()

    # ----------------------------------------------------------
    # PAINEL PRINCIPAL
    # ----------------------------------------------------------
    if 'meta_df' not in st.session_state:
        st.info("Insira um GSE ID e clique em **Fetch Data** para começar.")
        return

    current_mode = st.session_state.get('mode', 'Microarray')
    meta = st.session_state['meta_df'].copy()

    with st.expander("📊 Metadata Explorer", expanded=False):
        st.dataframe(meta, use_container_width=True)

    def make_label(row):
        parts = [str(row[c]) for c in label_cols if c in row.index]
        return " | ".join(parts) if parts else str(row.get('Accession', ''))

    meta['display_label'] = meta.apply(make_label, axis=1)
    all_labels = meta['display_label'].tolist()
    label_to_gsm = dict(zip(meta['display_label'], meta['Accession']))

    # ----------------------------------------------------------
    # GRUPOS
    # ----------------------------------------------------------
    st.subheader("🛠️ Group Management")
    groups = st.session_state['groups']

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        field_key = f"group_name_field_{st.session_state['group_field_key']}"
        new_g = st.text_input(
            "New Group Name:",
            key=field_key,
            placeholder="Ex: Control, Treatment...",
        )
    with col_btn:
        st.write(""); st.write("")
        add_clicked = st.button("➕ Add Group", use_container_width=True)

    if add_clicked and new_g.strip():
        if new_g.strip() not in groups:
            groups[new_g.strip()] = []
            st.session_state['groups'] = groups
        st.session_state['group_field_key'] += 1
        st.rerun()

    updated_groups = {}
    cols_g = st.columns(2)
    for i, (g_name, g_samples) in enumerate(list(groups.items())):
        already_taken = all_assigned_samples(groups, exclude=g_name)
        available = [lab for lab in all_labels if lab not in already_taken]
        valid_default = [s for s in g_samples if s in available]

        with cols_g[i % 2]:
            st.markdown(f"**Group: {g_name}**")
            sel = st.multiselect(
                "Select Samples:",
                available,
                default=valid_default,
                key=f"s_{g_name}",
            )
            updated_groups[g_name] = sel
            if st.button(f"🗑️ Remove {g_name}", key=f"del_{g_name}"):
                del st.session_state['groups'][g_name]
                st.rerun()

    st.session_state['groups'] = updated_groups

    # ----------------------------------------------------------
    # UPLOAD MANUAL
    # ----------------------------------------------------------
    df_current = st.session_state.get('df')
    matrix_empty = df_current is None or (
        isinstance(df_current, pd.DataFrame) and df_current.empty
    )

    if matrix_empty:
        st.warning(
            "📦 Nenhuma matriz de expressão encontrada automaticamente. "
            "Faça upload do arquivo de counts (.txt.gz / .tsv.gz / .tsv / .txt / .csv)."
        )
        up_file = st.file_uploader("Upload counts file", type=["gz", "tsv", "txt", "csv"])
        if up_file:
            try:
                raw = up_file.read()
                df_up = _parse_matrix_bytes(raw)
                if df_up is None:
                    st.error("Não foi possível parsear o arquivo. Verifique se é TSV/CSV com genes nas linhas e amostras nas colunas.")
                else:
                    # Tenta sincronizar com GSMs
                    gsm_order = st.session_state.get('gsm_order', [])
                    if gsm_order:
                        df_up, sync_msg = _sync_suppl_columns_with_gsms(df_up, gsm_order)
                        st.info(f"Sincronização: {sync_msg}")

                    # Detecta e mapeia IDs
                    id_type = detect_index_type(df_up.index.tolist())
                    st.session_state['id_type'] = id_type
                    with st.spinner(f"🔗 Mapeando IDs ({id_type}) → Gene Symbol..."):
                        mapping, msg = get_gene_mapping_rnaseq(
                            tuple(df_up.index.astype(str).tolist()), id_type
                        )
                    st.session_state['mapping'] = mapping
                    st.session_state['mapping_msg'] = msg
                    st.session_state['df'] = df_up
                    st.session_state['matrix_source'] = "Upload manual"
                    st.success(f"✅ {df_up.shape[0]} genes × {df_up.shape[1]} amostras. {msg}")
                    st.rerun()
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")
        return

    df_matrix = st.session_state['df']
    matrix_cols = set(df_matrix.columns.astype(str))

    # ----------------------------------------------------------
    # ANÁLISE
    # ----------------------------------------------------------
    if len(updated_groups) >= 2:
        st.divider()
        c1, c2 = st.columns(2)
        group_names = list(updated_groups.keys())
        ref_g = c1.selectbox("Referência (Controle):", group_names)
        test_g = c2.selectbox("Teste:", [g for g in group_names if g != ref_g])

        if st.button("🔥 Run Analysis", use_container_width=True):
            with st.spinner("Analisando..."):
                c_ref = [
                    label_to_gsm[lab]
                    for lab in updated_groups[ref_g]
                    if label_to_gsm.get(lab) in matrix_cols
                ]
                c_test = [
                    label_to_gsm[lab]
                    for lab in updated_groups[test_g]
                    if label_to_gsm.get(lab) in matrix_cols
                ]

                if not c_ref or not c_test:
                    st.error("❌ Nenhuma amostra válida encontrada nos grupos.")
                    with st.expander("🔍 Diagnóstico"):
                        st.write("**Colunas da matriz (primeiras 10):**", list(df_matrix.columns[:10]))
                        st.write("**GSMs — Ref:**", [label_to_gsm.get(l) for l in updated_groups[ref_g]])
                        st.write("**GSMs — Test:**", [label_to_gsm.get(l) for l in updated_groups[test_g]])
                else:
                    data_vals = df_matrix[c_ref + c_test].values.astype(np.float32)

                    if current_mode == "Microarray":
                        data_norm_vals = quantile_normalize(data_vals)
                    else:
                        data_norm_vals = np.log2(data_vals + 1.0)

                    data_norm = pd.DataFrame(
                        data_norm_vals,
                        columns=c_ref + c_test,
                        index=df_matrix.index,
                    )
                    m1 = data_norm[c_ref].values
                    m2 = data_norm[c_test].values

                    if current_mode == "Microarray" and use_limma:
                        p_l, f_l = [], []
                        for row in range(len(data_norm)):
                            y = np.concatenate([m1[row], m2[row]])
                            x = sm.add_constant(
                                np.concatenate([np.zeros(len(c_ref)), np.ones(len(c_test))])
                            )
                            mod = sm.OLS(y, x).fit()
                            p_l.append(mod.pvalues[1])
                            f_l.append(mod.params[1])
                        pvals = np.array(p_l)
                        lfc = np.array(f_l)
                    else:
                        lfc = np.nanmean(m2, axis=1) - np.nanmean(m1, axis=1)
                        pvals = stats.ttest_ind(
                            m2, m1, axis=1, equal_var=False, nan_policy='omit'
                        ).pvalue

                    res = pd.DataFrame({
                        'Probe_ID': df_matrix.index.astype(str),
                        'Log2FC': lfc,
                        'PValue': pvals,
                    }).dropna()

                    mapping = st.session_state.get('mapping')
                    if mapping is not None:
                        res = res.merge(mapping, on='Probe_ID', how='left')
                        res['Symbol'] = (
                            res['Symbol']
                            .replace(['nan', '---', ' ', 'None'], np.nan)
                            .fillna(res['Probe_ID'])
                        )
                    else:
                        res['Symbol'] = res['Probe_ID']

                    st.session_state.update({
                        'res': res,
                        'norm_df': data_norm,
                        'analysis_done': True,
                        'rn': ref_g,
                        'tn': test_g,
                    })
                    gc.collect()
                    st.rerun()

    # ----------------------------------------------------------
    # RESULTADOS
    # ----------------------------------------------------------
    if st.session_state.get('analysis_done'):
        res = st.session_state['res'].copy()
        df_norm = st.session_state['norm_df']

        if gene_area:
            s_list = [g.strip().upper() for g in gene_area.split('\n') if g.strip()]
            res = res[res['Symbol'].str.upper().isin(s_list)]

        df_diff = (
            res[(res['PValue'] < p_thr) & (res['Log2FC'].abs() >= fc_thr)]
            .sort_values('Log2FC', key=abs, ascending=False)
        )

        if df_diff.empty:
            st.info("Nenhum DEG encontrado com os thresholds atuais.")
        else:
            st.subheader(f"Results: {st.session_state['tn']} vs {st.session_state['rn']}")

            top30 = df_diff.head(30).copy()
            top30['Symbol'] = top30['Symbol'].astype(str)
            valid_ids = [idx for idx in top30['Probe_ID'].values if idx in df_norm.index]
            if valid_ids:
                h_mat = df_norm.loc[valid_ids].values
                h_z = (h_mat - np.mean(h_mat, axis=1, keepdims=True)) / (
                    np.std(h_mat, axis=1, keepdims=True) + 1e-9
                )
                fig_h = px.imshow(
                    h_z,
                    y=top30.iloc[:len(valid_ids)]['Symbol'].tolist(),
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                )
                fig_h.update_yaxes(type='category')
                st.plotly_chart(fig_h, use_container_width=True)

            st.divider()
            c_a, c_u, c_d = st.columns(3)

            f_a = px.bar(top30, x='Symbol', y='Log2FC', color='Log2FC',
                         color_continuous_scale='Greens', title="Top DEGs")
            f_a.update_xaxes(type='category')
            c_a.plotly_chart(f_a, use_container_width=True)

            up_g = df_diff[df_diff['Log2FC'] > 0].head(max_plot).copy()
            up_g['Symbol'] = up_g['Symbol'].astype(str)
            f_u = px.bar(up_g, x='Symbol', y='Log2FC', color='Log2FC',
                         color_continuous_scale='Blues', title="UP-regulated")
            f_u.update_xaxes(type='category')
            c_u.plotly_chart(f_u, use_container_width=True)

            dw_g = df_diff[df_diff['Log2FC'] < 0].head(max_plot).copy()
            dw_g['Symbol'] = dw_g['Symbol'].astype(str)
            f_d = px.bar(dw_g, x='Symbol', y='Log2FC', color='Log2FC',
                         color_continuous_scale='Reds_r', title="DOWN-regulated")
            f_d.update_xaxes(type='category')
            c_d.plotly_chart(f_d, use_container_width=True)

            st.subheader("Detailed Results Table")
            st.dataframe(
                df_diff[['Symbol', 'Log2FC', 'PValue']].rename(columns={'Symbol': 'Gene Symbol'}),
                use_container_width=True,
            )

    # ----------------------------------------------------------
    # DEBUG
    # ----------------------------------------------------------
    st.divider()
    with st.expander("🕵️ Debug", expanded=False):
        c_db1, c_db2 = st.columns(2)
        with c_db1:
            if current_mode == "Microarray" and st.session_state.get('raw_gpl') is not None:
                st.subheader("Raw GPL Mapping")
                st.dataframe(st.session_state['raw_gpl'], use_container_width=True)
            elif current_mode == "RNASeq":
                mapping = st.session_state.get('mapping')
                if mapping is not None:
                    st.subheader(f"Mapping ({st.session_state.get('id_type', '?')} → Symbol)")
                    st.dataframe(mapping.head(50), use_container_width=True)
                else:
                    st.info("Nenhum mapeamento disponível.")
        with c_db2:
            if st.session_state.get('res') is not None:
                st.subheader("Merge Result (primeiros 20)")
                st.dataframe(st.session_state['res'].head(20), use_container_width=True)
            df_m = st.session_state.get('df')
            if df_m is not None:
                st.subheader("Colunas da Matriz (primeiras 15)")
                st.write(list(df_m.columns[:15]))
                st.caption(f"Shape: {df_m.shape}")


if __name__ == '__main__':
    run_app()