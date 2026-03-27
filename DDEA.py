import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import requests
import gzip
import io
import os
from Bio import Entrez
import statsmodels.api as sm

# Identificação para evitar erro 401
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
Entrez.email = ""

# --- 1. FUNÇÕES TÉCNICAS REFORMULADAS (DIVISÃO DE TAREFAS) ---

def quantile_normalize(df_values):
    """Normalização por quantis otimizada (float32 para economizar RAM)."""
    sorted_df = np.sort(df_values.astype(np.float32), axis=0)
    rank_mean = sorted_df.mean(axis=1)
    indices = np.argsort(df_values, axis=0)
    norm_mat = np.empty_like(df_values, dtype=np.float32)
    for i in range(df_values.shape[1]):
        norm_mat[indices[:, i], i] = rank_mean
    return norm_mat

@st.cache_data(show_spinner=False)
def get_gene_mapping_stepped(gse_id, probe_ids):
    """Passo 2: Mapeamento cirúrgico de símbolos."""
    try:
        search = Entrez.read(Entrez.esearch(db="gds", term=f"{gse_id}[ACCN]"))
        uid = search["IdList"][0]
        record = Entrez.read(Entrez.esummary(db="gds", id=uid))
        gpl_id = record[0]['GPL']
        gpl_num = gpl_id.replace("GPL", "")
        prefix = f"GPL{gpl_num[:-3]}nnn" if len(gpl_num) > 3 else "GPLnnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/GPL{gpl_id}/soft/GPL{gpl_id}_family.soft.gz"
        
        r = requests.get(url, stream=True, headers=HEADERS)
        with gzip.open(r.raw, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('!platform_table_begin'): break
            
            # Carrega apenas as colunas necessárias para não dar 404
            map_df = pd.read_csv(f, sep='\t', comment='!', low_memory=True, on_bad_lines='skip')
            target_cols = ['Gene.Symbol', 'Gene Symbol', 'GENE_SYMBOL', 'Symbol']
            sym_col = next((c for c in map_df.columns if any(k == c or k == c.upper() for k in target_cols)), None)
            
            if sym_col:
                map_df = map_df[map_df['ID'].astype(str).isin(probe_ids)]
                map_df[sym_col] = map_df[sym_col].astype(str).apply(lambda x: x.split(' /// ')[0])
                return map_df[['ID', sym_col]].rename(columns={'ID': 'Probe_ID', sym_col: 'Symbol'})
        return None
    except: return None

@st.cache_data
def get_geo_matrix_stepped(gse_id):
    """Passo 1: Download da matriz e metadados com busca robusta de cabeçalho."""
    num = gse_id.replace("GSE", "")
    prefix = f"GSE{num[:-3]}nnn" if len(num) > 3 else "GSEnnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
    
    try:
        r = requests.get(url, stream=True, headers=HEADERS)
        if r.status_code != 200: return None, None, f"Erro {r.status_code}: Acesso negado."
        
        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            titles, gsms, data_buffer = [], [], []
            found_data = False
            
            for line in f:
                if line.startswith('!Sample_title'): titles = [t.strip().replace('"', '') for t in line.split('\t')[1:]]
                if line.startswith('!Sample_geo_accession'): gsms = [t.strip().replace('"', '') for t in line.split('\t')[1:]]
                # Busca robusta pelo início dos dados
                if not line.startswith('!') and len(line.split('\t')) > 1:
                    data_buffer.append(line)
                    found_data = True
                    # Lê o restante do arquivo de uma vez para o buffer
                    data_buffer.extend(f.readlines())
                    break
            
            if not found_data: return None, None, "Não foi possível encontrar a tabela de dados no arquivo."

            # Processa a matriz
            df = pd.read_csv(io.StringIO("".join(data_buffer)), sep='\t')
            # Mapeia GSM para Título Original
            name_map = dict(zip(gsms, titles)) if gsms and titles else {}
            
            # Limpeza e conversão para float32 (Economia de RAM)
            new_cols = [df.columns[0]] + [name_map.get(c.replace('"', ''), c.replace('"', '')) for c in df.columns[1:]]
            df.columns = new_cols
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
            
            return df.dropna(subset=df.columns[1:], how='all'), df.columns[1:].tolist(), None
    except Exception as e: return None, None, str(e)

# --- 2. INTERFACE STREAMLIT ---

def run_app():
    st.set_page_config(layout="wide", page_title="DDEA Ultra Complete")
    st.title("Diagonal Differential Expression Alley 🧬")

    if 'groups' not in st.session_state: st.session_state['groups'] = {}

    with st.sidebar:
        if os.path.exists("DDEA-small.png"): st.image("DDEA-small.png", use_container_width=True)
        st.header("1. Experiment Setup")
        exp_mode = st.radio("Experiment Type:", ["Microarray", "RNASeq"])
        gse_input = st.text_input("GSE ID:", value="GSE78093")
        
        st.divider()
        st.header("2. Gene Search Filter")
        gene_txt = st.text_area("Specific Genes (1 per line):")
        uploaded_txt = st.file_uploader("Upload .txt list", type="txt")
        
        st.divider()
        st.header("3. DDEA Parameters")
        use_limma = st.checkbox("Linear Model (Limma-like)", value=False)
        p_thr = st.slider("P-value threshold:", 0.001, 0.10, 0.05, format="%.3f")
        fc_thr = st.slider("Min Abs Log2(FC):", 0.0, 5.0, 0.0, step=0.1)
        max_plot = st.number_input("Max genes to plot:", value=20)
        
        fetch_btn = st.button("🔍 Fetch from GEO", use_container_width=True)

    # Coleta de genes para filtro customizado
    c_genes = [g.strip().upper() for g in gene_txt.split('\n') if g.strip()]
    if uploaded_txt:
        c_genes += [line.decode('utf-8').strip().upper() for line in uploaded_txt.readlines()]

    if fetch_btn and gse_input:
        with st.spinner("🚀 Passo 1: Baixando Matriz e Títulos Reais..."):
            df, titles, err = get_geo_matrix_stepped(gse_input)
            if not err:
                st.session_state['df'], st.session_state['titles'] = df, titles
                with st.spinner("🧬 Passo 2: Mapeando Gene Symbols (RAM-Safe)..."):
                    st.session_state['mapping'] = get_gene_mapping_stepped(gse_input, df.iloc[:, 0].astype(str).tolist())
                st.session_state['groups'] = {"Control": titles[:len(titles)//2], "Experimental": titles[len(titles)//2:]}
                st.session_state['analysis_done'] = False
            else: st.error(err)

    if 'df' in st.session_state:
        df, titles = st.session_state['df'], st.session_state['titles']
        
        st.subheader("🛠️ Group Management")
        groups = st.session_state['groups']
        new_g = st.text_input("Add New Group Name:")
        if st.button("➕ Add Group") and new_g:
            groups[new_g] = []; st.session_state['groups'] = groups; st.rerun()

        # Configuração de múltiplos grupos
        updated_groups = {}
        cols_g = st.columns(3)
        for i, (g_name, g_samples) in enumerate(list(groups.items())):
            with cols_g[i % 3]:
                n_name = st.text_input(f"Name:", value=g_name, key=f"n_{g_name}")
                sel = st.multiselect(f"Samples:", titles, default=g_samples, key=f"s_{g_name}")
                updated_groups[n_name] = sel
                if st.button(f"🗑️ Remove {n_name}", key=f"del_{g_name}"):
                    del st.session_state['groups'][g_name]; st.rerun()
        st.session_state['groups'] = updated_groups

        st.divider()
        if len(updated_groups) >= 2:
            c1, c2 = st.columns(2)
            ref_g = c1.selectbox("Control (Ref):", list(updated_groups.keys()))
            test_g = c2.selectbox("Experimental:", [g for g in updated_groups.keys() if g != ref_g])

            if st.button("🔥 Run Analysis (Normalized)", use_container_width=True):
                with st.spinner("Calculando..."):
                    c_ref, c_test = updated_groups[ref_g], updated_groups[test_g]
                    data_vals = df[c_ref + c_test].values
                    if np.nanmax(data_vals) > 50: data_vals = np.log2(data_vals + 1.0)
                    data_norm = pd.DataFrame(quantile_normalize(data_vals), columns=c_ref + c_test, index=df.index)
                    
                    m1, m2 = data_norm[c_ref].values, data_norm[c_test].values
                    if use_limma:
                        p_l, f_l = [], []
                        for row in range(len(data_norm)):
                            y, x = np.concatenate([m1[row], m2[row]]), sm.add_constant(np.concatenate([np.zeros(len(c_ref)), np.ones(len(c_test))]))
                            res_sm = sm.OLS(y, x).fit()
                            p_l.append(res_sm.pvalues[1]); f_l.append(res_sm.params[1])
                        pvals, lfc = np.array(p_l), np.array(f_l)
                    else:
                        lfc = np.nanmean(m2, axis=1) - np.nanmean(m1, axis=1)
                        pvals = stats.ttest_ind(m2, m1, axis=1, equal_var=False, nan_policy='omit').pvalue
                    
                    res = pd.DataFrame({'Probe_ID': df.iloc[:, 0].astype(str), 'Log2FC': lfc, 'PValue': pvals, 'neg_log10_p': -np.log10(pvals + 1e-10)}).dropna()
                    if st.session_state.get('mapping') is not None:
                        res = res.merge(st.session_state['mapping'], on='Probe_ID', how='left')
                        res['Symbol'] = res['Symbol'].replace(['nan', '---', ' '], np.nan).fillna(res['Probe_ID'])
                    else: res['Symbol'] = res['Probe_ID']
                    st.session_state.update({'res': res, 'norm_df': data_norm, 'analysis_done': True, 'rn': ref_g, 'tn': test_g})

        # --- EXIBIÇÃO REATIVA ---
        if st.session_state.get('analysis_done'):
            res = st.session_state['res']
            if c_genes: res = res[res['Symbol'].str.upper().isin(c_genes)]
            df_diff = res[(res['PValue'] < p_thr) & (res['Log2FC'].abs() >= fc_thr)].sort_values('Log2FC', key=abs, ascending=False)

            if not df_diff.empty:
                st.subheader(f"1. Heatmap: {st.session_state['tn']} vs {st.session_state['rn']}")
                top30 = df_diff.head(30)
                # Amostras originais no Heatmap
                h_mat = st.session_state['norm_df'].loc[top30.index].values
                h_z = (h_mat - np.mean(h_mat, axis=1, keepdims=True)) / (np.std(h_mat, axis=1, keepdims=True) + 1e-9)
                st.plotly_chart(px.imshow(h_z, y=top30['Symbol'].tolist(), color_continuous_scale='RdBu_r', height=750, aspect="auto"), use_container_width=True)

                st.divider()
                st.subheader("2. Expression Patterns")
                c1, c2, c3 = st.columns(3)
                c1.plotly_chart(px.bar(df_diff.head(max_plot), x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Viridis', title="Top DEGs"), use_container_width=True)
                c2.plotly_chart(px.bar(df_diff[df_diff['Log2FC']>0].head(max_plot), x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Blues', title="UP Genes"), use_container_width=True)
                c3.plotly_chart(px.bar(df_diff[df_diff['Log2FC']<0].head(max_plot), x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Reds_r', title="DOWN Genes"), use_container_width=True)

                st.subheader("3. Detailed Results Table")
                df_f = df_diff[['Symbol', 'Log2FC', 'neg_log10_p']].rename(columns={'Symbol': 'Gene Symbol', 'Log2FC': 'Log2(FoldChange)', 'neg_log10_p': '-log10(Pvalue)'})
                st.dataframe(df_f, use_container_width=True)

if __name__ == '__main__': run_app()
