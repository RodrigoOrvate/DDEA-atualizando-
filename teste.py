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

HEADERS = {'User-Agent': 'DDEA/1.0 (Streamlit App; +https://github.com)'}
Entrez.email = "ddea.tool@example.com"


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

def _strip_ensembl_version(ensembl_id: str) -> str:
    """Remove sufixo de versão de Ensembl IDs: ENSG00000129824.11 → ENSG00000129824"""
    return ensembl_id.split('.')[0] if '.' in ensembl_id else ensembl_id


@st.cache_data(show_spinner=False)
def get_gene_mapping_rnaseq(index_ids: tuple, id_type: str):
    """
    Converte IDs para Gene Symbol conforme o tipo detectado.
    - entrez  → MyGene.info POST com field=symbol
    - ensembl → MyGene.info POST com field=symbol, scopes=ensembl.gene
                (remove sufixo de versão, ex: ENSG00000129824.11 → ENSG00000129824)
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
        # Para entrez, o ID original == ID limpo
        original_to_clean = {i: i for i in ids_clean}
    else:
        # Ensembl: remove versão para consulta, mas mantém mapeamento de volta
        ids_raw = [str(i).strip() for i in index_ids if str(i).strip().startswith('ENS')]
        scope = "ensembl.gene"
        # Mapa: ID original (com versão) → ID sem versão
        original_to_clean = {i: _strip_ensembl_version(i) for i in ids_raw}
        # Mapa reverso: ID sem versão → ID original (primeiro encontrado)
        clean_to_original = {}
        for orig, clean in original_to_clean.items():
            if clean not in clean_to_original:
                clean_to_original[clean] = orig
        ids_clean = list(clean_to_original.keys())

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
                if id_type == 'ensembl' and query_id in clean_to_original:
                    # Mapeia de volta para o ID original com versão
                    original_id = clean_to_original[query_id]
                    results.append({"Probe_ID": original_id, "Symbol": symbol})
                else:
                    results.append({"Probe_ID": query_id, "Symbol": symbol})
        except Exception as e:
            return None, f"MyGene.info erro: {e}"

    if not results:
        return None, "MyGene.info não retornou resultados."

    mapping_df = pd.DataFrame(results).drop_duplicates('Probe_ID')
    mapping_df["Symbol"] = mapping_df["Symbol"].fillna(mapping_df["Probe_ID"])
    # Para Ensembl, adicionar IDs que ficaram sem mapeamento (versões duplicadas etc.)
    if id_type == 'ensembl':
        mapped_set = set(mapping_df['Probe_ID'])
        for orig, clean in original_to_clean.items():
            if orig not in mapped_set:
                # Tentar encontrar o símbolo pelo ID sem versão
                match = mapping_df[mapping_df['Probe_ID'].apply(_strip_ensembl_version) == clean]
                if not match.empty:
                    sym = match.iloc[0]['Symbol']
                    results.append({"Probe_ID": orig, "Symbol": sym})
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

def _parse_series_type(series_type_raw: str) -> str:
    """
    Interpreta o campo !Series_type da Series Matrix e retorna
    'Microarray', 'RNASeq' ou 'unknown'.
    """
    s = series_type_raw.lower()
    if 'high throughput sequencing' in s:
        return 'RNASeq'
    if 'array' in s:
        return 'Microarray'
    if 'sequencing' in s:
        return 'RNASeq'
    return 'unknown'


def _try_series_matrix(gse_id):
    """
    Baixa a series_matrix.
    Retorna (df_expression_or_None, meta_df, gsms, gsm_order, detected_type).
    gsm_order é a lista ordenada de GSMs conforme o cabeçalho.
    detected_type é 'Microarray', 'RNASeq' ou 'unknown'.
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
            series_type_raw = ""
            df = pd.DataFrame()
            for line in f:
                if line.startswith('!Series_type'):
                    series_type_raw = line.split('\t')[1].strip().replace('"', '') if '\t' in line else ""
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

            detected_type = _parse_series_type(series_type_raw)

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
                return None, meta_df, gsms, gsm_order, detected_type

            df = df.set_index(0)
            df.index = df.index.astype(str).str.strip().str.replace('"', '')
            col_rename = {col: gsm_order[i] for i, col in enumerate(df.columns) if i < len(gsm_order)}
            df.rename(columns=col_rename, inplace=True)

            num_cols = df.select_dtypes(include=[np.number]).shape[1]
            if num_cols < 2:
                return None, meta_df, gsms, gsm_order, detected_type

            df = df.select_dtypes(include=[np.number])
            return df, meta_df, gsms, gsm_order, detected_type

    except Exception as e:
        return None, None, None, [], 'unknown'


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


def _try_extract_symbol_mapping_from_suppl(gse_id, ensembl_ids, log_cb=None):
    """
    Tenta extrair mapeamento Ensembl→Symbol dos arquivos suplementares do GEO.
    Muitos datasets RNA-Seq incluem arquivos com colunas como 'gene_name', 'Symbol', etc.
    Também tenta usar o índice se for gene symbol e há coluna Ensembl.
    Retorna (mapping_df, msg) ou (None, msg).
    """
    urls = sorted(_list_supplementary_urls(gse_id), key=_score_suppl_file, reverse=True)
    if not urls:
        return None, "Nenhum arquivo suplementar encontrado."

    # Colunas que indicam gene symbol
    symbol_keywords = [
        'gene_name', 'genename', 'gene.name', 'gene symbol', 'gene.symbol',
        'genesymbol', 'symbol', 'gene_symbol', 'hgnc_symbol', 'external_gene_name',
        'external_gene_id', 'name', 'genes'
    ]
    # Colunas que indicam Ensembl ID
    ensembl_keywords = [
        'ensembl', 'ensg', 'gene_id', 'geneid', 'ensembl_gene_id',
        'ensembl_id', 'gene.id'
    ]

    ensembl_set_stripped = {_strip_ensembl_version(e) for e in ensembl_ids}

    for url in urls:
        fname = url.split('/')[-1]
        try:
            if log_cb:
                log_cb(f"Buscando mapping em: `{fname}`")
            r = requests.get(url, timeout=60, headers=HEADERS)
            if r.status_code != 200:
                continue

            try:
                content = gzip.decompress(r.content)
            except Exception:
                content = r.content

            # Tenta ler só as primeiras linhas para detectar colunas
            for sep in ['\t', ',']:
                try:
                    df_test = pd.read_csv(io.BytesIO(content), sep=sep, nrows=5, low_memory=False)
                    if df_test.shape[1] < 2:
                        continue

                    cols_lower = {c: c.lower().strip().replace('"', '') for c in df_test.columns}

                    # Encontra coluna de symbol
                    sym_col = None
                    for orig, low in cols_lower.items():
                        if any(kw == low for kw in symbol_keywords):
                            sym_col = orig
                            break
                    if sym_col is None:
                        for orig, low in cols_lower.items():
                            if any(kw in low for kw in symbol_keywords):
                                sym_col = orig
                                break

                    # Encontra coluna de Ensembl ID
                    ens_col = None
                    for orig, low in cols_lower.items():
                        if any(kw == low for kw in ensembl_keywords):
                            ens_col = orig
                            break
                    if ens_col is None:
                        for orig, low in cols_lower.items():
                            if any(kw in low for kw in ensembl_keywords):
                                ens_col = orig
                                break

                    if sym_col is None:
                        # Checar se o índice (primeira coluna após read_csv com index_col=0)
                        # pode ser gene symbol e outra coluna é Ensembl
                        continue

                    # Ler o arquivo completo com as colunas de interesse
                    df_full = pd.read_csv(io.BytesIO(content), sep=sep, low_memory=False)
                    df_full.columns = [str(c).strip().replace('"', '') for c in df_full.columns]

                    if ens_col and ens_col in df_full.columns and sym_col in df_full.columns:
                        # Temos ambas as colunas!
                        map_df = df_full[[ens_col, sym_col]].copy()
                        map_df.columns = ['Ensembl_ID', 'Symbol']
                        map_df['Ensembl_ID'] = map_df['Ensembl_ID'].astype(str).str.strip()
                        map_df['Symbol'] = map_df['Symbol'].astype(str).str.strip()
                        # Verificar se os Ensembl IDs batem com os do dataset
                        map_stripped = map_df['Ensembl_ID'].apply(_strip_ensembl_version)
                        overlap = map_stripped.isin(ensembl_set_stripped).sum()
                        if overlap >= 10:
                            # Mapear de volta usando o ID original do dataset
                            # Cria dict: ensembl_sem_versão → symbol
                            ens_to_sym = dict(zip(map_stripped, map_df['Symbol']))
                            result_rows = []
                            for eid in ensembl_ids:
                                eid_clean = _strip_ensembl_version(str(eid))
                                sym = ens_to_sym.get(eid_clean)
                                if sym and sym not in ('nan', '', 'None', 'NA', '---'):
                                    result_rows.append({'Probe_ID': str(eid), 'Symbol': sym})
                            if result_rows:
                                result_df = pd.DataFrame(result_rows).drop_duplicates('Probe_ID')
                                n_mapped = len(result_df)
                                return result_df, f"{n_mapped}/{len(ensembl_ids)} IDs mapeados via arquivo suplementar `{fname}`."

                    elif sym_col in df_full.columns:
                        # Tem coluna de symbol, vamos checar o índice
                        df_idx = pd.read_csv(io.BytesIO(content), sep=sep, index_col=0, low_memory=False)
                        df_idx.index = df_idx.index.astype(str).str.strip()
                        idx_stripped = {_strip_ensembl_version(i) for i in df_idx.index[:50] if 'ENS' in str(i)}
                        if len(idx_stripped.intersection(ensembl_set_stripped)) >= 5:
                            # O índice é Ensembl, e temos coluna de symbol
                            sym_values = df_idx[sym_col].astype(str).str.strip()
                            result_rows = []
                            for eid in ensembl_ids:
                                eid_str = str(eid).strip()
                                if eid_str in df_idx.index:
                                    sym = sym_values.loc[eid_str]
                                    if isinstance(sym, pd.Series):
                                        sym = sym.iloc[0]
                                    if sym and sym not in ('nan', '', 'None', 'NA', '---'):
                                        result_rows.append({'Probe_ID': eid_str, 'Symbol': sym})
                            if not result_rows:
                                # Tentar sem versão
                                idx_to_sym = {}
                                for idx_val, sym_val in zip(df_idx.index, sym_values):
                                    clean = _strip_ensembl_version(idx_val)
                                    if clean not in idx_to_sym:
                                        idx_to_sym[clean] = sym_val
                                for eid in ensembl_ids:
                                    eid_clean = _strip_ensembl_version(str(eid))
                                    sym = idx_to_sym.get(eid_clean)
                                    if sym and sym not in ('nan', '', 'None', 'NA', '---'):
                                        result_rows.append({'Probe_ID': str(eid), 'Symbol': sym})
                            if result_rows:
                                result_df = pd.DataFrame(result_rows).drop_duplicates('Probe_ID')
                                return result_df, f"{len(result_df)}/{len(ensembl_ids)} IDs mapeados via arquivo suplementar `{fname}`."

                except Exception:
                    continue
        except Exception:
            continue

    return None, "Nenhum arquivo suplementar continha mapeamento Ensembl→Symbol."


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
    Retorna (df_matrix, meta_df, gsms, gsm_order, source, error, detected_type).
    detected_type: 'Microarray', 'RNASeq' ou 'unknown' — detectado a partir dos metadados GEO.
    """
    if log_cb:
        log_cb("Buscando metadados e Series Matrix...")

    df_matrix, meta_df, gsms, gsm_order, detected_type = _try_series_matrix(gse_id)

    if meta_df is None:
        return None, None, None, [], None, "Falha ao acessar o GEO. Verifique o GSE ID.", 'unknown'

    if mode == "Microarray":
        if df_matrix is not None:
            return df_matrix, meta_df, gsms, gsm_order, "Series Matrix", None, detected_type
        return None, meta_df, gsms, gsm_order, None, None, detected_type

    # RNA-Seq: cascata
    if df_matrix is not None and df_matrix.shape[1] >= 2:
        return df_matrix, meta_df, gsms, gsm_order, "Series Matrix", None, detected_type

    if log_cb:
        log_cb("Series Matrix sem dados de expressão. Buscando arquivos suplementares...")
    df_suppl = _try_supplementary(gse_id, log_cb=log_cb)
    if df_suppl is not None:
        df_synced, sync_method = _sync_suppl_columns_with_gsms(df_suppl, gsm_order)
        if log_cb:
            log_cb(f"Sincronização de colunas: {sync_method}")
        return df_synced, meta_df, gsms, gsm_order, f"Supplementary TXT ({sync_method})", None, detected_type

    if log_cb:
        log_cb("Suplementares não encontrados. Tentando NCBI-generated counts...")
    df_ncbi = _try_ncbi_generated(gse_id, log_cb=log_cb)
    if df_ncbi is not None:
        df_synced, sync_method = _sync_suppl_columns_with_gsms(df_ncbi, gsm_order)
        return df_synced, meta_df, gsms, gsm_order, f"NCBI-generated ({sync_method})", None, detected_type

    if log_cb:
        log_cb("Nenhuma fonte automática encontrou dados. Upload manual necessário.")
    return None, meta_df, gsms, gsm_order, None, None, detected_type


# ============================================================
# HELPERS DE ESTADO
# ============================================================

def reset_analysis_state():
    for k in ['df', 'meta_df', 'res', 'analysis_done', 'mapping', 'mapping_msg',
              'raw_gpl', 'mode', 'norm_df', 'rn', 'tn', 'gse_id',
              'gsms', 'gsm_order', 'matrix_source', 'id_type', 'detected_type']:
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
        gse_input = st.text_input("GSE ID:", placeholder="Ex: GSE117769")
        fetch_btn = st.button("🚀 Fetch Data", use_container_width=True)

        if 'meta_df' in st.session_state:
            st.divider()
            det = st.session_state.get('detected_type', 'unknown')
            if det != 'unknown':
                tipo_label = "🧬 RNA-Seq" if det == 'RNASeq' else "🔬 Microarray"
                st.caption(f"🏷️ Tecnologia detectada: **{tipo_label}**")
                if st.session_state.get('mode') and st.session_state['mode'] != det:
                    st.warning(
                        f"⚠️ Tipo selecionado (**{st.session_state['mode']}**) "
                        f"difere do detectado (**{det}**). "
                        f"Considere trocar para **{det}** e clicar em Fetch novamente."
                    )
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
            df, meta_df, gsms, gsm_order, source, err, detected_type = get_geo_full_data(
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
            st.session_state['detected_type'] = detected_type

            # Aviso se o tipo selecionado difere do detectado
            if detected_type != 'unknown' and mode != detected_type:
                det_label = "RNA-Seq" if detected_type == 'RNASeq' else "Microarray"
                sel_label = "RNA-Seq" if mode == 'RNASeq' else "Microarray"
                st.warning(
                    f"⚠️ **Tecnologia incorreta!** Você selecionou **{sel_label}**, mas o GEO indica "
                    f"que este dataset é **{det_label}** (`!Series_type`). "
                    f"Troque para **{det_label}** na sidebar e clique em **Fetch Data** novamente "
                    f"para obter resultados corretos."
                )

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
                    mapping = None
                    msg = ""

                    if id_type == 'ensembl':
                        # Estratégia 1: Tentar extrair mapeamento dos suplementares
                        with st.spinner("🔍 Buscando mapeamento nos arquivos suplementares..."):
                            mapping, msg = _try_extract_symbol_mapping_from_suppl(
                                gse_input.strip(),
                                df.index.astype(str).tolist(),
                                log_cb=log_cb
                            )

                    if mapping is None:
                        # Estratégia 2: MyGene.info (com strip de versão para Ensembl)
                        with st.spinner(f"🔗 Mapeando IDs ({id_type}) → Gene Symbol via MyGene.info..."):
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
    # ANÁLISE MULTI-GRUPO
    # ----------------------------------------------------------
    if len(updated_groups) >= 2:
        st.divider()
        MAX_GROUPS = 4
        group_names = list(updated_groups.keys())
        
        if len(group_names) > MAX_GROUPS:
            st.error(f"⚠️ Limite de segurança: O sistema suporta no máximo {MAX_GROUPS} grupos para evitar instabilidade.")
        else:
            import itertools
            # Gera todas as combinações possíveis: (A, B), (A, C), (B, C)...
            comparisons = list(itertools.combinations(group_names, 2))
            
            st.subheader("🧪 Configuração da Análise")
            st.info(f"Detectados {len(group_names)} grupos. Serão realizadas {len(comparisons)} comparações par a par + 1 análise global (ANOVA).")
            
            # Opção de visualização rápida para o usuário
            selected_comp = st.selectbox(
                "Selecione a comparação principal para exibição nos gráficos:",
                [f"{c[1]} vs {c[0]}" for c in comparisons] + (["Global (ANOVA)"] if len(group_names) > 2 else [])
            )

            if st.button("🔥 Run Multi-Group Analysis", use_container_width=True):
                all_results = {}
                full_norm_df = pd.DataFrame()
                
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 1. LOOP DE COMPARAÇÕES PAR A PAR
                for idx, (g_ref, g_test) in enumerate(comparisons):
                    comp_name = f"{g_test}_vs_{g_ref}"
                    status_text.text(f"Processando: {comp_name}...")
                    
                    c_ref = [label_to_gsm[lab] for lab in updated_groups[g_ref] if label_to_gsm.get(lab) in matrix_cols]
                    c_test = [label_to_gsm[lab] for lab in updated_groups[g_test] if label_to_gsm.get(lab) in matrix_cols]

                    if not c_ref or not c_test:
                        continue

                    # Extração e Normalização
                    data_vals = df_matrix[c_ref + c_test].values.astype(np.float32)
                    if current_mode == "Microarray":
                        data_norm_vals = quantile_normalize(data_vals)
                    else:
                        data_norm_vals = np.log2(data_vals + 1.0)

                    data_norm = pd.DataFrame(data_norm_vals, columns=c_ref + c_test, index=df_matrix.index)
                    
                    # Estatística Par a Par
                    m1 = data_norm[c_ref].values
                    m2 = data_norm[c_test].values

                    if current_mode == "Microarray" and use_limma:
                        p_l, f_l = [], []
                        for row in range(len(data_norm)):
                            y = np.concatenate([m1[row], m2[row]])
                            x = sm.add_constant(np.concatenate([np.zeros(len(c_ref)), np.ones(len(c_test))]))
                            mod = sm.OLS(y, x).fit()
                            p_l.append(mod.pvalues[1])
                            f_l.append(mod.params[1])
                        pvals, lfc = np.array(p_l), np.array(f_l)
                    else:
                        lfc = np.nanmean(m2, axis=1) - np.nanmean(m1, axis=1)
                        pvals = stats.ttest_ind(m2, m1, axis=1, equal_var=False, nan_policy='omit').pvalue

                    res = pd.DataFrame({
                        'Probe_ID': df_matrix.index.astype(str),
                        'Log2FC': lfc,
                        'PValue': pvals,
                    }).dropna()

                    # Mapeamento de Símbolos
                    mapping = st.session_state.get('mapping')
                    if mapping is not None:
                        res = res.merge(mapping, on='Probe_ID', how='left')
                        res['Symbol'] = res['Symbol'].replace(['nan', '---', ' ', 'None'], np.nan).fillna(res['Probe_ID'])
                    else:
                        res['Symbol'] = res['Probe_ID']

                    all_results[comp_name] = res
                    progress_bar.progress((idx + 1) / (len(comparisons) + 1))

                # 2. ANÁLISE GLOBAL (ANOVA) - Apenas se houver 3+ grupos
                if len(group_names) > 2:
                    status_text.text("Executando Teste Global (ANOVA)...")
                    all_samples = []
                    group_labels = []
                    for g in group_names:
                        samps = [label_to_gsm[lab] for lab in updated_groups[g] if label_to_gsm.get(lab) in matrix_cols]
                        all_samples.extend(samps)
                        group_labels.append(df_matrix[samps].values)

                    # ANOVA One-way por gene
                    f_stat, p_anova = stats.f_oneway(*group_labels, axis=1)
                    
                    anova_res = pd.DataFrame({
                        'Probe_ID': df_matrix.index.astype(str),
                        'Log2FC': f_stat, # Usamos o F-stat no lugar do LFC para visualização de magnitude
                        'PValue': p_anova,
                    }).dropna()
                    
                    if mapping is not None:
                        anova_res = anova_res.merge(mapping, on='Probe_ID', how='left')
                        anova_res['Symbol'] = anova_res['Symbol'].fillna(anova_res['Probe_ID'])
                    else:
                        anova_res['Symbol'] = anova_res['Probe_ID']
                        
                    all_results["Global (ANOVA)"] = anova_res

                # Atualização do Estado
                # Para manter compatibilidade com seus gráficos, definimos 'res' como a comparação selecionada
                key_res = selected_comp.replace(" ", "_") if "Global" not in selected_comp else "Global (ANOVA)"
                
                st.session_state.update({
                    'all_results': all_results,
                    'res': all_results.get(selected_comp.replace(" ", "_"), all_results.get("Global (ANOVA)")),
                    'analysis_done': True,
                    'rn': selected_comp.split(' vs ')[1] if ' vs ' in selected_comp else "Vários",
                    'tn': selected_comp.split(' vs ')[0] if ' vs ' in selected_comp else "Global",
                    'norm_df': df_matrix[all_samples] if len(group_names)>2 else data_norm # Simplificado
                })
                
                progress_bar.empty()
                status_text.success("Análise concluída!")
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
