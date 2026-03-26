import streamlit as st
import pandas as pd
import plotly.express as px
import re
from Bio import Entrez

def run_gene_expression_analysis():
    st.set_page_config(layout="wide", page_title="DDEA - Bioinformática UFRN")
    st.title("Diagonal Differential Expression Alley 🧬")

    tab1, tab2 = st.tabs(["Analysis Dashboard", "Documentation"])

    with tab1:
        with st.sidebar:
            st.header("⚙️ Configurações de API")
            # Variável dinâmica para o e-mail do usuário
            user_email = st.text_input(
                "E-mail para o NCBI:", 
                placeholder="seu.email@ufrn.edu.br",
                help="O NCBI exige um e-mail para identificar as requisições ao banco de dados GEO."
            )
            
            st.markdown("---")
            st.header("1. Pesquisa no NCBI GEO")
            search_query = st.text_input("Busque por Doença ou GSE ID:", placeholder="Ex: Alzheimer")
            
            if search_query:
                if not user_email:
                    st.warning("⚠️ Por favor, insira seu e-mail acima para habilitar a busca no NCBI.")
                else:
                    Entrez.email = user_email # Define o e-mail dinamicamente
                    with st.spinner("Buscando no GEO..."):
                        try:
                            # Busca no banco GDS (Genome Data Bank)
                            handle = Entrez.esearch(db="gds", term=f"{search_query}[DataSet]", retmax=5)
                            record = Entrez.read(handle)
                            ids = record["IdList"]
                            
                            if ids:
                                st.success(f"Encontrados {len(ids)} estudos!")
                                summaries = Entrez.esummary(db="gds", id=",".join(ids))
                                options = {f"{s['Accession']}: {s['Title'][:50]}...": s['Accession'] for s in summaries}
                                selected_study = st.selectbox("Selecione o estudo:", options.keys())
                                
                                st.markdown(f"🔗 [Acessar página do {options[selected_study]}](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={options[selected_study]})")
                            else:
                                st.warning("Nenhum estudo encontrado.")
                        except Exception as e:
                            st.error(f"Erro na API do NCBI: {e}")

            st.markdown("---")
            st.header("2. Upload de Dados")
            uploaded_file = st.file_uploader("Upload seu arquivo DEG (.tsv)", type=["tsv"])
            
            st.subheader("Filtros DDEA")
            p_threshold = st.number_input("P-value threshold:", value=0.05, step=0.001, format="%.4f")
            fc_threshold = st.number_input("Min abs Log2(FC):", value=0.0, step=0.1)
            max_genes = st.number_input("Máximo de genes no gráfico (0=todos):", value=20, step=1)

        # --- Lógica de Processamento ---
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep='\t')
            
            # Detecção automática de colunas do DDEA original
            gene_col = 'Symbol'
            logFC_pattern = r'log2\(fold change\)\((.*)\)'
            p_pattern = r'-log10\(Pvalue\)\((.*)\)'
            
            found_fc = found_p = comp_string = None
            for col in df.columns:
                m_fc = re.match(logFC_pattern, col)
                if m_fc:
                    found_fc = col
                    comp_string = m_fc.group(1)
                if re.match(p_pattern, col):
                    found_p = col

            if found_fc and found_p:
                st.info(f"Comparação detectada: **{comp_string}**")
                
                # Cálculo: P-valor = $10^{-\text{-log10(Pvalue)}}$
                df['P_val'] = 10**(-df[found_p])
                
                # Filtragem
                mask = (df['P_val'] < p_threshold) & (df[found_fc].abs() >= fc_threshold)
                df_filtered = df[mask].copy()
                df_filtered['abs_FC'] = df_filtered[found_fc].abs()
                df_filtered = df_filtered.sort_values(by='abs_FC', ascending=False)

                if max_genes > 0:
                    df_filtered = df_filtered.head(max_genes)

                # Gráfico Plotly Express
                fig = px.bar(df_filtered, x=gene_col, y=found_fc, color=found_fc,
                             title=f"Análise DDEA - {comp_string}",
                             color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Tabela de Genes Filtrados")
                st.dataframe(df_filtered[[gene_col, found_fc, 'P_val']])
            else:
                st.error("Formato de colunas não reconhecido (use o padrão GEO2R).")
        else:
            st.info("Aguardando upload de arquivo para iniciar análise.")

    with tab2:
        st.markdown("### Documentação DDEA")
        st.write("Esta ferramenta facilita a visualização de DEGs em modelos animais e doenças.")

if __name__ == '__main__':
    run_gene_expression_analysis()