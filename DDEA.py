import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import requests
import gzip
import io
import os
import gc
from Bio import Entrez
import statsmodels.api as sm

# Configuração técnica NCBI
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleBio/1.0'}
Entrez.email = "rodrigo.arruda@ufrn.edu.br"

# --- 1. MOTORES DE PERFORMANCE E MAPEAMENTO ---

def quantile_normalize(df_values):
    """Normalização por quantis em float32 (Segurança contra 404/Killing)."""
    if df_values.size == 0: return df_values
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

@st.cache_data(show_spinner=False)
def get_gene_mapping(gse_id):
    """Mapeia Probe_ID para Gene.Symbol (GEO2R Fidelity)."""
    try:
        search_handle = Entrez.esearch(db="gds", term=f"{gse_id}[ACCN]")
        uid = Entrez.read(search_handle)["IdList"][0]
        record = Entrez.read(Entrez.esummary(db="gds", id=uid))
        gpl_id = record[0]['GPL']
        prefix = f"GPL{gpl_id[:-3]}nnn" if len(gpl_id) > 3 else "GPLnnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/GPL{gpl_id}/soft/GPL{gpl_id}_family.soft.gz"
        r = requests.get(url, stream=True, headers=HEADERS)
        with gzip.open(r.raw, 'rt', encoding='utf-8', errors='ignore') as f:
            table_lines = []
            in_table = False
            for line in f:
                if line.startswith('!platform_table_begin'): in_table = True; continue
                if line.startswith('!platform_table_end'): break
                if in_table: table_lines.append(line)
                if len(table_lines) > 150000: break
            map_df = pd.read_csv(io.StringIO("".join(table_lines)), sep='\t', low_memory=False)
            target = ['Gene.Symbol', 'Gene Symbol', 'GENE_SYMBOL', 'Symbol', 'SYMBOL']
            col = next((c for c in map_df.columns if any(k == c or k == c.upper() for k in target)), None)
            if col:
                map_df['ID'] = map_df['ID'].astype(str).str.strip().str.replace('"', '')
                map_df[col] = map_df[col].astype(str).apply(lambda x: x.split(' /// ')[0])
                return map_df[['ID', col]].rename(columns={'ID': 'Probe_ID', col: 'Symbol'})
        return None
    except: return None

@st.cache_data(show_spinner=False)
def try_auto_fetch_counts(gse_id):
    """Tenta baixar o arquivo suplementar de RNA-seq (BETA) do NCBI."""
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:5]}nnn/{gse_id}/suppl/{gse_id}_raw_counts_GRCh38.p13_NCBI.tsv.gz"
    try:
        r = requests.get(url, headers=HEADERS, stream=True)
        if r.status_code == 200:
            df = pd.read_csv(io.BytesIO(r.content), sep='\t', index_col=0, compression='gzip')
            df.columns = [str(c).replace('"', '').strip() for c in df.columns]
            return df, None
        return None, "Auto-fetch falhou."
    except Exception as e: return None, str(e)

@st.cache_data(show_spinner=False)
def get_geo_matrix_robust(gse_id):
    """Baixa matriz e títulos reais das amostras."""
    num = gse_id.replace("GSE", "")
    prefix = f"GSE{num[:-3]}nnn" if len(num) > 3 else "GSEnnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
    try:
        r = requests.get(url, stream=True, headers=HEADERS)
        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            titles, gsms = [], []
            df = pd.DataFrame()
            for line in f:
                if line.startswith('!Sample_title'): titles = [t.strip().replace('"', '') for t in line.split('\t')[1:]]
                if line.startswith('!Sample_geo_accession'): gsms = [t.strip().replace('"', '') for t in line.split('\t')[1:]]
                if line.startswith('ID_REF') or line.startswith('"ID_REF"'):
                    df = pd.read_csv(f, sep='\t', header=None, low_memory=True)
                    break
            num_cols = df.shape[1] - 1 if not df.empty else len(gsms)
            final_names = titles if (len(titles) == num_cols) else gsms
            if not df.empty:
                df.columns = ['ID_REF'] + final_names[:df.shape[1]-1]
                df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip().str.replace('"', '')
                for col in df.columns[1:]: df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
            gc.collect()
            return df, final_names, gsms, None
    except Exception as e: return None, None, None, str(e)

# --- 2. INTERFACE COMPLETA ---

def run_app():
    st.set_page_config(layout="wide", page_title="DDEA Complete Ultra")
    st.title("Diagonal Differential Expression Alley 🧬")

    if 'groups' not in st.session_state: st.session_state['groups'] = {}

    with st.sidebar:
        st.header("1. Experiment Setup")
        exp_mode = st.radio("Experiment Type:", ["Microarray", "RNASeq"])
        gse_input = st.text_input("GSE ID:", value="GSE78093")
        gene_area = st.text_area("Genes to Highlight:")
        
        st.divider()
        st.header("2. Parameters")
        p_thr = st.slider("P-value threshold:", 0.001, 0.10, 0.05, format="%.3f")
        fc_thr = st.slider("Min Abs Log2FC:", 0.0, 5.0, 0.0, step=0.1)
        use_limma = st.checkbox("Use Linear Model (Limma-like)", value=False)
        max_plot = st.number_input("Max genes in bar charts:", value=20)
        fetch_btn = st.button("🔍 Fetch from GEO", use_container_width=True)

    if fetch_btn and gse_input:
        for k in ['df', 'titles', 'res', 'analysis_done', 'mapping']: st.session_state.pop(k, None)
        gc.collect()
        with st.spinner("🚀 Downloading and Parsing..."):
            df, titles, gsms, err = get_geo_matrix_robust(gse_input)
            if not err:
                st.session_state['df'], st.session_state['titles'], st.session_state['gsms'] = df, titles, gsms
                st.session_state['mapping'] = get_gene_mapping(gse_input)
                st.session_state['groups'] = {"Control": titles[:len(titles)//2], "Experimental": titles[len(titles)//2:]}
                st.session_state['analysis_done'] = False
                if exp_mode == "RNASeq" and (df is None or df.empty):
                    counts_df, c_err = try_auto_fetch_counts(gse_input)
                    if counts_df is not None:
                        st.session_state['df'] = counts_df
                        st.success("✅ Arquivo de contagens carregado automaticamente!")
            else: st.error(err)

    if 'titles' in st.session_state:
        df, titles = st.session_state['df'], st.session_state['titles']
        
        if df is None or df.empty:
            st.warning("📦 Matriz de contagens não encontrada. Faça o upload manual.")
            up_file = st.file_uploader("Upload GSE_raw_counts.tsv.gz", type=["gz", "tsv"])
            if up_file:
                st.session_state['df'] = pd.read_csv(up_file, sep='\t', index_col=0)
                st.session_state['df'].columns = [str(c).replace('"', '').strip() for c in st.session_state['df'].columns]
                st.rerun()

        st.subheader("🛠️ Group Management")
        groups = st.session_state['groups']
        new_g = st.text_input("New Group Name:")
        if st.button("➕ Add Group") and new_g:
            groups[new_g] = []; st.session_state['groups'] = groups; st.rerun()

        updated_groups = {}
        cols_g = st.columns(3)
        for i, (g_name, g_samples) in enumerate(list(groups.items())):
            with cols_g[i % 3]:
                st.markdown(f"**Group: {g_name}**")
                sel = st.multiselect(f"Samples:", titles, default=[s for s in g_samples if s in titles], key=f"s_{g_name}")
                updated_groups[g_name] = sel
                if st.button(f"🗑️ Remove", key=f"del_{g_name}"):
                    del st.session_state['groups'][g_name]; st.rerun()
        st.session_state['groups'] = updated_groups

        if st.session_state['df'] is not None and not st.session_state['df'].empty and len(updated_groups) >= 2:
            st.divider()
            c1, c2 = st.columns(2)
            ref_g = c1.selectbox("Reference:", list(updated_groups.keys()))
            test_g = c2.selectbox("Test:", [g for g in updated_groups.keys() if g != ref_g])

            if st.button("🔥 Run Analysis", use_container_width=True):
                with st.spinner("Analisando..."):
                    df_run = st.session_state['df']
                    gsms = st.session_state['gsms']
                    t_to_g = dict(zip(titles, gsms))
                    
                    # --- DETETIVE DE COLUNAS (NOVA LÓGICA) ---
                    available_cols = list(df_run.columns)
                    def find_col(target_title):
                        target_gsm = t_to_g.get(target_title, "")
                        # Tenta pelo GSM Exato
                        if target_gsm in available_cols: return target_gsm
                        # Tenta se o GSM está contido no nome da coluna (ex: "GSM123_Sample")
                        for col in available_cols:
                            if target_gsm and target_gsm in col: return col
                        # Tenta pelo Título Exato
                        if target_title in available_cols: return target_title
                        return None

                    c_ref = [find_col(t) for t in updated_groups[ref_g] if find_col(t)]
                    c_test = [find_col(t) for t in updated_groups[test_g] if find_col(t)]
                    
                    if len(c_ref) == 0 or len(c_test) == 0:
                        st.error(f"❌ Erro de Sincronia: Amostras não encontradas no arquivo.")
                        st.info(f"Colunas detectadas no arquivo: {available_cols[:5]}...")
                        st.info(f"GSMs procurados: {[t_to_g.get(t) for t in updated_groups[ref_g][:3]]}...")
                    else:
                        data_vals = df_run[c_ref + c_test].values
                        if data_vals.size > 0:
                            if np.nanmax(data_vals) > 50: data_vals = np.log2(data_vals.astype(float) + 1.0)
                            data_norm = pd.DataFrame(quantile_normalize(data_vals), columns=c_ref + c_test, index=df_run.index)
                            m1, m2 = data_norm[c_ref].values, data_norm[c_test].values
                            lfc = np.nanmean(m2, axis=1) - np.nanmean(m1, axis=1)
                            pvals = stats.ttest_ind(m2, m1, axis=1, equal_var=False, nan_policy='omit').pvalue
                            res = pd.DataFrame({'Probe_ID': df_run.index.astype(str), 'Log2FC': lfc, 'PValue': pvals}).dropna()
                            if st.session_state.get('mapping') is not None:
                                res = res.merge(st.session_state['mapping'], on='Probe_ID', how='left')
                                res['Symbol'] = res['Symbol'].replace(['nan', '---', ' '], np.nan).fillna(res['Probe_ID'])
                            else: res['Symbol'] = res['Probe_ID']
                            st.session_state.update({'res': res, 'norm_df': data_norm, 'analysis_done': True, 'rn': ref_g, 'tn': test_g})
                        gc.collect()

        if st.session_state.get('analysis_done'):
            res = st.session_state['res']
            if gene_area: res = res[res['Symbol'].str.upper().isin([g.strip().upper() for g in gene_area.split('\n')])]
            df_diff = res[(res['PValue'] < p_thr) & (res['Log2FC'].abs() >= fc_thr)].sort_values('Log2FC', key=abs, ascending=False)
            if not df_diff.empty:
                st.subheader(f"Results: {st.session_state['tn']} vs {st.session_state['rn']}")
                top30 = df_diff.head(30)
                h_mat = st.session_state['norm_df'].loc[top30['Probe_ID']].values
                h_z = (h_mat - np.mean(h_mat, axis=1, keepdims=True)) / (np.std(h_mat, axis=1, keepdims=True) + 1e-9)
                st.plotly_chart(px.imshow(h_z, y=top30['Symbol'].tolist(), color_continuous_scale='RdBu_r', aspect="auto"), use_container_width=True)
                c_a, c_u, c_d = st.columns(3)
                c_a.plotly_chart(px.bar(df_diff.head(max_plot), x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Greens', title="Top DEGs"), use_container_width=True)
                c_u.plotly_chart(px.bar(df_diff[df_diff['Log2FC']>0].head(max_plot), x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Blues', title="UP"), use_container_width=True)
                c_d.plotly_chart(px.bar(df_diff[df_diff['Log2FC']<0].head(max_plot), x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Reds_r', title="DOWN"), use_container_width=True)

if __name__ == '__main__': run_app()