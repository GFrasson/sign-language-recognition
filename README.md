# Sign Language Recognition

Este projeto é um sistema de Reconhecimento de Língua de Sinais (Sign Language Recognition) baseado em aprendizado profundo (Deep Learning) e extração de características geométricas, desenvolvido em Python usando TensorFlow/Keras e MediaPipe.

## 📌 Sobre o Projeto

O objetivo deste projeto é classificar sinais de linguagem de sinais a partir de vídeos. Para lidar com a complexidade e a variação da execução de sinais por diferentes pessoas (sinalizadores), o projeto emprega uma abordagem robusta de extração de características físicas e arquiteturas neurais temporais (LSTM). Além disso, utiliza uma abordagem de classificação hierárquica (com "Modelos Especialistas") para desambiguar sinais visualmente semelhantes.

## 🚀 Principais Funcionalidades

- **Extração Avançada de Features**: Utiliza o MediaPipe para extrair pontos de referência (landmarks) esqueléticos, faciais e manuais.
- **Features Geométricas Complexas**: Calcula distâncias, ângulos articulares e a normal da palma das mãos, melhorando a precisão do reconhecimento.
- **Features Temporais (Velocidade)**: Suporte opcional a características de velocidade e aceleração temporal dos movimentos.
- **Arquitetura Hierárquica / Especialistas**: Treina e utiliza modelos LSTM dedicados exclusivamente à desambiguação de classes conflitantes (ex: classes 4 e 7, 16 e 17).
- **Aumento de Dados (Data Augmentation)**: Processamento e amostragem aleatória de frames para evitar *overfitting* e aumentar a robustez do modelo.
- **Validação Cruzada Rigorosa**: Metodologia de *Leave-Two-Signalers-Out Cross-Validation* para testar a generalização real do modelo em dados não vistos (novos indivíduos).

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Linguagem**: Python 3
- **Deep Learning**: TensorFlow / Keras (Redes Neurais Recorrentes LSTM)
- **Visão Computacional e Landmarks**: MediaPipe
- **Processamento de Dados**: NumPy, scikit-learn
- **Paralelismo**: `concurrent.futures` para otimizar o processamento de vídeo
- **Visualização**: Matplotlib (Matrizes de confusão e histórico de treinamento)

## 📁 Estrutura de Diretórios (Resumo)

- `src/main.py`: Ponto de entrada principal. Lida com a configuração de argumentos, divisão dos folds e orquestração do treinamento.
- `src/video_processing.py`: Script focado na orquestração da leitura de vídeos e extração das features em paralelo.
- `src/geometric_features.py`: Cálculos matemáticos detalhados dos ângulos e distâncias a partir dos landmarks brutos.
- `src/entities/`: Classes de domínio (POOs).
  - `Dataset.py`: Gerenciamento e separação do conjunto de dados.
  - `Model.py` / `HierarchicalModel.py` / `SpecialistModel.py`: Definições das redes neurais LSTM gerais e os modelos especialistas.
  - `Settings.py`: Definições constantes, número de features e parâmetros da rede.

## ⚙️ Como Executar

O projeto possui uma interface via linha de comando parametrizável (`argparse`). 

Instale as dependências (recomendado usar um ambiente virtual):
```bash
pip install -r requirements.txt
```

**Treinamento Padrão:**
```bash
python src/main.py
```

**Todos os argumentos disponíveis:**
- `--use-specialist-4-7`: Habilita o modelo especialista para diferenciar as classes 4 e 7.
- `--use-specialist-16-17`: Habilita o modelo especialista para diferenciar as classes 16 e 17.
- `--train-specialist-only <classe>`: Treina *apenas* o modelo especialista para a classe gatilho informada (ex: 4 ou 16). Filtra o conjunto de dados para usar apenas as classes relevantes a este especialista.
- `--use-velocity`: Inclui features de velocidade/aceleração (dinâmicas temporais) para o modelo geral.
- `--specialist-only-velocity`: Faz com que os modelos especialistas utilizem *apenas* as features de velocidade (requer que ao menos uma flag de especialista esteja ativa).
- `--legacy-features`: Retorna ao uso das 88 features originais de geometria (ao invés das 126 estendidas na nova versão).
- `--balance-specialist-data`: Usa apenas 50% dos dados das classes de especialistas durante o treinamento do Modelo Geral, com o objetivo de manter o balanceamento das classes.
- `--unroll-lstm`: Desenrola (unroll) a arquitetura da rede LSTM, o que pode ajudar a evitar erros do CuDNN (NVIDIA) com modelos muito grandes.
- `--start-fold <indice>`: Índice do *fold* de validação cruzada onde a execução deve iniciar (útil para retomar treinamentos interrompidos).
- `--resume-folder <caminho>`: Caminho de uma pasta de experimento existente para retomar o salvamento de resultados exatamente a partir do ponto de parada.
- `--evaluate-only`: Pula a fase de treinamento e realiza apenas a avaliação dos dados (requer a flag `--load-models-from`).
- `--load-models-from <caminho>`: Caminho para a pasta de um experimento contendo os modelos já treinados que serão carregados para a avaliação.

## 📊 Resultados e Modelos

O projeto cria pastas timestamped (ex: `YYYYMMDD_HHMMSS_lstm_...`) dentro do diretório configurado, nas quais salva:
- Os modelos treinados em formato H5/Keras.
- Os gráficos da história do treinamento (perda e acurácia).
- Matrizes de confusão por *fold* e um global da validação cruzada.
- Arquivo texto (`run_settings.txt`) documentando todos os parâmetros utilizados na execução, visando a completa **reprodutibilidade**.

## 📝 Notas de Desenvolvimento
O script de execução usa técnicas para garantir o determinismo e reprodutibilidade (fixação das *seeds* do NumPy, Python, e TensorFlow). Pode utilizar a aceleração de GPU caso o ambiente CUDA esteja corretamente configurado, habilitando crescimento de memória sob demanda.
