<h1 align="center">
  🤗 EmoClassifier
</h1>

<p align="center">
  <strong>Classificador de Emoções em Texto com BERT</strong>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://huggingface.co/">
    <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-ffca28.svg?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/Scikit--Learn-1.2+-f7931e.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit Learn">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
  </a>
</p>

<p align="center">
  🚀 <em>Simples, Treinável e Extensível.</em>
</p>

<br>

## 📖 Sobre o Projeto

O **EmoClassifier** é um pipeline robusto e eficiente para **classificação de emoções em textos em português**. Ele utiliza o poder do **BERT** (especificamente o modelo `neuralmind/bert-base-portuguese-cased`) para entender nuances linguísticas e categorizar frases em diversas emoções.

Este projeto é ideal para:
*   📊 **Análise de Sentimento e Emoção** em feedbacks de clientes.
*   📱 **Monitoramento de Redes Sociais**.
*   🤖 **Chatbots e Assistentes Virtuais** empáticos.
*   🔬 **Pesquisa em NLP** (Processamento de Linguagem Natural).

### Emoções Suportadas
O modelo é configurado para identificar as seguintes classes (baseado no modelo de Plutchik ou similar):
`Neutro`, `Alegria`, `Tristeza`, `Raiva`, `Medo`, `Nojo`, `Surpresa`, `Confiança`, `Antecipação`.

---

## ✨ Funcionalidades

*   **Fine-Tuning de BERT**: Ajuste fino de modelos Transformer pré-treinados para a tarefa específica de classificação.
*   **Treinamento Eficiente**: Suporte a **Mixed Precision Training (AMP)** para maior velocidade e menor uso de memória.
*   **Pipeline Completo**: Desde o carregamento dos dados (CSV) até a avaliação do modelo.
*   **Métricas Detalhadas**: Cálculo automático de Acurácia, Precision, Recall e F1-Score.
*   **Early Stopping**: Interrompe o treinamento automaticamente quando o modelo para de melhorar, evitando overfitting.
*   **Reprodutibilidade**: Sementes (seeds) fixadas para garantir resultados consistentes.

---

## 📁 Estrutura do Repositório

```text
EmoClassifier/
├── data/                  # 📂 Armazene seus datasets (CSVs) aqui
├── models/                # 💾 Modelos treinados são salvos aqui
├── training_pipeline.py   # ⚙️ Script principal de treinamento e validação
├── requirements.txt       # 📦 Lista de dependências do projeto
├── .gitignore             # 🙈 Arquivos ignorados pelo Git
├── LICENSE                # 📜 Licença de uso (MIT)
└── README.md              # 📘 Esta documentação
```

---

## 🚀 Instalação

Siga os passos abaixo para configurar o ambiente de desenvolvimento:

### 1. Clone o repositório
```bash
git clone https://github.com/ramoneirao/EmoClassifier.git
cd EmoClassifier
```

### 2. Crie um ambiente virtual (Recomendado)
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

---

## 🛠️ Como Usar

### 1. Preparação dos Dados
Crie um arquivo CSV (ex: `data/meu_dataset.csv`) contendo as colunas:
*   `texto`: O texto a ser classificado.
*   Colunas para cada emoção (One-Hot Encoding): `neutro`, `alegria`, `tristeza`, `raiva`, `medo`, `nojo`, `surpresa`, `confianca`, `antecipacao`.

> **Nota:** O pipeline converte automaticamente o formato One-Hot para índices de classe para o treinamento.

### 2. Treinando o Modelo
Execute o script `training_pipeline.py`. Você pode ajustar os parâmetros diretamente no bloco `__main__` do arquivo ou implementar argumentos via linha de comando (CLI) futuramente.

```bash
python training_pipeline.py
```

O script irá:
1.  Carregar os dados.
2.  Dividir em treino e validação.
3.  Treinar o modelo BERT por `N` épocas.
4.  Salvar o melhor modelo na pasta `models/`.
5.  Exibir métricas de desempenho.

### 3. Exemplo de Inferência (Código)
Para usar o modelo treinado em seus próprios scripts:

```python
import torch
from training_pipeline import BERTTrainer

# Configuração
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "models/best_model.pth"

# Inicializar o treinador (que contém o modelo)
trainer = BERTTrainer(num_classes=9)
trainer.load_best_model(model_path)

# Textos para classificar
textos = [
    "Estou muito feliz com essa notícia maravilhosa!",
    "Que dia terrível, tudo deu errado.",
    "Não sei o que esperar do futuro."
]

# Predição
preds = trainer.predict(textos)

# Mapeamento (exemplo)
mapa_emocoes = {0: 'Neutro', 1: 'Alegria', 2: 'Tristeza', 3: 'Raiva', 4: 'Medo', 
                5: 'Nojo', 6: 'Surpresa', 7: 'Confiança', 8: 'Antecipação'}

resultados = [mapa_emocoes[p] for p in preds]
print(resultados)
# Saída esperada: ['Alegria', 'Tristeza', 'Antecipação'] (exemplo)
```

---

## 💡 Roadmap e Melhorias Futuras

*   [ ] Adicionar suporte a argumentos via linha de comando (`argparse`).
*   [ ] Criar um script dedicado para inferência (`predict.py`).
*   [ ] Implementar logging avançado (WandB, MLflow).
*   [ ] Suportar outros modelos pré-treinados (RoBERTa, DistilBERT).
*   [ ] Criar uma API simples (FastAPI) para servir o modelo.

---

## 🤝 Contribuição

Contribuições são muito bem-vindas! Sinta-se à vontade para abrir **Issues** para relatar bugs ou sugerir melhorias, e **Pull Requests** para enviar código.

1.  Faça um Fork do projeto.
2.  Crie uma Branch para sua feature (`git checkout -b feature/MinhaFeature`).
3.  Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`).
4.  Push para a Branch (`git push origin feature/MinhaFeature`).
5.  Abra um Pull Request.

---

## 📄 Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  Feito com 💜 por <a href="https://github.com/ramoneirao">Ramon</a>
</p>