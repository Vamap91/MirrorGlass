# Sistema de Detecção de Fraudes em Imagens

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

![Carglass Automotiva](https://via.placeholder.com/150x50?text=Carglass)

## 📋 Sobre o Projeto

Este sistema utiliza técnicas avançadas de visão computacional para detectar imagens duplicadas ou altamente semelhantes em pedidos automotivos, identificando possíveis fraudes onde a mesma imagem é utilizada para gerar múltiplos pedidos.

### 🎯 Objetivos

- Automatizar a detecção de fraudes em pedidos com imagens duplicadas
- Identificar duplicatas mesmo com pequenas alterações (recortes, ajustes, etc.)
- Acelerar o processo de auditoria com alertas visuais
- Gerar relatórios detalhados de possíveis fraudes

## ✨ Funcionalidades

- **Upload de múltiplas imagens** para análise simultânea
- **Extração de características visuais** utilizando algoritmos state-of-the-art (SIFT)
- **Comparação inteligente** entre todas as imagens do acervo
- **Interface visual** para revisão rápida de possíveis duplicatas
- **Geração de relatórios** em formato CSV

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Framework para interface web
- **OpenCV/SIFT**: Algoritmo avançado para detecção de características visuais 
- **scikit-image**: Biblioteca para processamento de imagens
- **Python**: Linguagem de programação base

## 🚀 Como Usar

### Instalação Local

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/detector-duplicatas.git
cd detector-duplicatas
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run app.py
```

### Deploy no Streamlit Cloud

1. Forke este repositório no GitHub
2. Acesse [Streamlit Cloud](https://streamlit.io/cloud)
3. Crie um novo app apontando para seu fork
4. Defina `app.py` como arquivo principal

## 📈 Como Funciona

1. **Extração de características**: O sistema utiliza o algoritmo SIFT (Scale-Invariant Feature Transform) para extrair pontos-chave das imagens
2. **Correspondência de características**: Compara estes pontos-chave entre as imagens usando FLANN (Fast Library for Approximate Nearest Neighbors)
3. **Cálculo de similaridade**: Determina o grau de semelhança entre as imagens baseado nas correspondências encontradas
4. **Filtro por limiar**: Identifica como duplicatas as imagens com similaridade acima do limiar definido

## 📊 Interpretação dos Resultados

- **Similaridade 100%**: Imagens idênticas
- **Similaridade >90%**: Praticamente idênticas (recortadas ou com filtros)
- **Similaridade 70-90%**: Muito semelhantes (prováveis duplicatas)
- **Similaridade 50-70%**: Semelhantes (verificar manualmente)
- **Similaridade 30-50%**: Possivelmente relacionadas (verificar com atenção)
- **Similaridade <30%**: Provavelmente não são duplicatas

## 🔧 Ajustes e Configurações

O sistema permite ajustar:

- **Limiar de Similaridade**: Define a sensibilidade da detecção
- **Método de Detecção**: Escolha entre SIFT, SSIM ou combinação de ambos
- **Tamanho de Pré-processamento**: Ajusta o tamanho das imagens processadas

## 📝 Próximas Melhorias

- [ ] Integração com banco de dados para armazenamento permanente
- [ ] Processamento em lotes para grandes volumes
- [ ] Implementação de OCR para extração de texto nas imagens
- [ ] Painel administrativo para auditoria
- [ ] Alerta automático por email

## 👥 Desenvolvido para

**Carglass Automotiva**  
Projeto: Detecção de Fraudes em Pedidos com Imagens Duplicadas

## 📄 Licença

Este projeto é proprietário e de uso exclusivo da Carglass Automotiva.

---

© 2025 Carglass Automotiva - Todos os direitos reservados
