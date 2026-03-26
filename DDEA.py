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

# E-mail técnico interno
Entrez.email = "bioinfo.ufrn@gmail.com"

# --- 1. FUNÇÕES COM CACHE (PARA PERFORMANCE) ---

@st.cache_data
def get_gene_mapping(gse_id):
    """Busca mapeamento Probe -> Symbol."""
    try:
        h = Entrez.esummary(db="gds", id=gse_id.replace("GSE", ""))
        record = Entrez.read(h)
        gpl_id = record[0]['GPL']
        gpl_prefix = f"GPL{gpl_id[:-3]}nnn" if len(gpl_id) > 3 else "GPLnnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl_prefix}/GPL{gpl_id}/soft/GPL{gpl_id}_family.soft.gz"
        r = requests.get(url)
        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            for line in f:
                if line.startswith('!platform_table_begin'):
                    map_df = pd.read_csv(f, sep='\t', comment='!')
                    sym_col = next((c for c in map_df.columns if any(k in c.upper() for k in ['SYMBOL', 'GENE_SYMBOL'])), None)
                    if sym_col:
                        return map_df[['ID', sym_col]].rename(columns={'ID': 'Probe_ID', sym_col: 'Symbol'})
        return None
    except: return None

@st.cache_data
def get_series_matrix(gse_id):
    """Baixa a matriz bruta."""
    gse_prefix = gse_id[:5] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_prefix}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
    try:
        r = requests.get(url)
        if r.status_code != 200: return None, None, "GSE não encontrado."
        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            lines = f.readlines()
            sample_titles = [l.split('\t')[1:] for l in lines if l.startswith('!Sample_title')]
            titles = [t.strip().replace('"', '') for t in sample_titles[0]] if sample_titles else []
            data_start = next(i for i, line in enumerate(lines) if not line.startswith('!'))
            clean_data = [l for l in lines[data_start:] if not l.startswith('!') and l.strip()]
            df = pd.read_csv(io.StringIO("".join(clean_data)), sep='\t')
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna(subset=df.columns[1:], how='all'), titles, None
    except Exception as e: return None, None, str(e)

# --- 2. INTERFACE E LÓGICA ---

def run_app():
    st.set_page_config(layout="wide", page_title="DDEA Pro UFRN")
    st.title("Diagonal Differential Expression Alley 🧬")

    # Sidebar sempre visível
    with st.sidebar:
        if os.path.exists("DDEA-small.png"): st.image("DDEA-small.png", use_container_width=True)
        st.header("1. Data Retrieval")
        gse_id = st.text_input("GSE ID:", placeholder="Ex: GSE78093")
        
        # Botão para buscar dados (Reset no reload garantido aqui)
        fetch_btn = st.button("🔍 Fetch from GEO", use_container_width=True)
        
        st.divider()
        st.header("2. DDEA Thresholds")
        # Estes sliders são reativos: mudou aqui, mudam os gráficos lá embaixo
        p_thr = st.slider("P-value (α):", 0.001, 0.10, 0.05, format="%.3f")
        fc_thr = st.slider("Min Abs Log2(FC):", 0.0, 5.0, 1.0)
        max_plot = st.number_input("Max genes to plot:", value=20)

    # Lógica de Carregamento
    if fetch_btn and gse_id:
        with st.spinner("Downloading data..."):
            df, titles, err = get_series_matrix(gse_id)
            if not err:
                st.session_state['raw_df'] = df
                st.session_state['titles'] = titles
                st.session_state['mapping'] = get_gene_mapping(gse_id)
                st.session_state['analysis_done'] = False # Reseta a análise se buscar novo GSE
            else: st.error(err)

    # Se os dados estão carregados, mostramos a gestão de grupos
    if 'raw_df' in st.session_state:
        df, titles = st.session_state['raw_df'], st.session_state['titles']
        
        with st.expander("📄 Raw Matrix Preview (Head)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        st.subheader("🛠️ Group Management")
        c1, c2 = st.columns(2)
        with c1: 
            g1_name = st.text_input("Group 1 Name:", value="Control")
            g1_sel = st.multiselect(f"Samples for {g1_name}:", titles, default=titles[:3])
        with c2:
            g2_name = st.text_input("Group 2 Name:", value="Experimental")
            g2_sel = st.multiselect(f"Samples for {g2_name}:", titles, default=titles[3:])

        # BOTÃO PARA RODAR A ANÁLISE (Obrigatório clicar uma vez)
        if st.button("🔥 Run DDEA Analysis", use_container_width=True):
            if g1_sel and g2_sel:
                with st.spinner("Calculating Statistics..."):
                    ids_gsm = df.columns[1:]
                    cols1 = [ids_gsm[titles.index(t)] for t in g1_sel]
                    cols2 = [ids_gsm[titles.index(t)] for t in g2_sel]
                    
                    m1, m2 = df[cols1].values.astype(float), df[cols2].values.astype(float)
                    lfc = np.nanmean(m2, axis=1) - np.nanmean(m1, axis=1)
                    pvals = stats.ttest_ind(m2, m1, axis=1, equal_var=False, nan_policy='omit').pvalue
                    
                    res = pd.DataFrame({
                        'Probe_ID': df.iloc[:, 0].astype(str),
                        'Log2FC': lfc, 'PValue': pvals,
                        'neg_log10_p': -np.log10(pvals + 1e-10)
                    }).dropna()

                    if st.session_state.get('mapping') is not None:
                        res = res.merge(st.session_state['mapping'], on='Probe_ID', how='left')
                        res['Symbol'] = res['Symbol'].fillna(res['Probe_ID'])
                    else: res['Symbol'] = res['Probe_ID']
                    
                    st.session_state['res'] = res
                    st.session_state['analysis_done'] = True
            else: st.warning("Please select samples for both groups.")

        # --- EXIBIÇÃO REATIVA ---
        if st.session_state.get('analysis_done'):
            res = st.session_state['res']
            df_diff = res[(res['PValue'] < p_thr) & (res['Log2FC'].abs() >= fc_thr)].copy()
            df_diff = df_diff.sort_values(by='Log2FC', key=abs, ascending=False)

            st.divider()
            st.subheader(f"Results: {g2_name} vs {g1_name}")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Total Unique Genes", len(res))
            col_m2.metric("Differentially Expressed", len(df_diff))

            if not df_diff.empty:
                # 🌡️ HEATMAP
                st.subheader("1. Gene Expression Heatmap")
                top_30 = df_diff.head(30)
                h_data = df[df.iloc[:, 0].isin(top_30['Probe_ID'])]
                z_vals = (h_data[df.columns[1:]].values - np.nanmean(h_data[df.columns[1:]].values, axis=1, keepdims=True)) / np.nanstd(h_data[df.columns[1:]].values, axis=1, keepdims=True)
                
                fig_h = px.imshow(z_vals, x=titles, y=top_30['Symbol'].tolist(),
                                  color_continuous_scale='RdBu_r', aspect="auto", height=750)
                st.plotly_chart(fig_h, use_container_width=True)

                # 📊 BAR CHARTS
                st.divider()
                st.subheader("2. Expression Patterns")
                c_all, c_up, c_down = st.columns(3)
                with c_all:
                    st.write("**All DEGs**")
                    st.plotly_chart(px.bar(df_diff.head(max_plot), x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Viridis'), use_container_width=True)
                with c_up:
                    st.write("**Upregulated**")
                    up = df_diff[df_diff['Log2FC'] > 0].head(max_plot)
                    st.plotly_chart(px.bar(up, x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Blues'), use_container_width=True)
                with c_down:
                    st.write("**Downregulated**")
                    down = df_diff[df_diff['Log2FC'] < 0].head(max_plot)
                    st.plotly_chart(px.bar(down, x='Symbol', y='Log2FC', color='Log2FC', color_continuous_scale='Reds_r'), use_container_width=True)

                # 📋 TABELA FINAL
                st.divider()
                st.subheader("3. Detailed Results Table")
                df_disp = df_diff[['Symbol', 'Log2FC', 'neg_log10_p']].rename(columns={
                    'Symbol': 'Gene Symbol', 'Log2FC': 'Log2(FoldChange)', 'neg_log10_p': '-log10(Pvalue)'
                })
                st.dataframe(df_disp, use_container_width=True)
                
                tsv = df_disp.to_csv(index=False, sep='\t').encode('utf-8')
                st.download_button("📥 Download TSV", tsv, f"DDEA_{g2_name}_vs_{g1_name}.tsv", "text/tab-separated-values")
            else:
                st.warning("No genes passed the thresholds. Adjust the sliders in the sidebar.")

if __name__ == '__main__':
    run_app()