# Model Card — Churn Prediction Pipeline

**Projeto:** Tech Challenge Fase 1 — PosTech  
**Versão:** 1.0  
**Data:** Junho 2026  
**Autores:** Patrick Covre · Beatriz Ferrante · Cecilia Rocha · Daniel Lacerda · Josef Caique

---

## Descrição do Modelo

Pipeline de classificação binária para prever churn (cancelamento) de clientes de uma operadora de telecomunicações. O pipeline combina pré-processamento sklearn com o modelo campeão registrado no MLflow Model Registry, servido via API FastAPI.

**Tipo de tarefa:** Classificação binária (churn = 1 / não churn = 0)  
**Framework:** Scikit-Learn (pipeline de produção) · PyTorch (MLP experimental)  
**Tracking:** MLflow (parâmetros, métricas, artefatos, Model Registry)  
**Servido via:** FastAPI — `POST /predict`

---

## Uso Pretendido

### Uso recomendado

- Identificar clientes com alto risco de cancelamento para acionar equipes de retenção proativa.
- Priorizar contatos de CRM com base no score de probabilidade retornado pelo endpoint `/predict`.
- Geração de lista diária de clientes em risco via batch.

### Fora do escopo

- **Não** deve ser usado como único critério para cancelar benefícios ou serviços de clientes.
- **Não** deve ser aplicado a datasets de outros setores sem retreino.
- **Não** substitui análise humana para casos de alto impacto individual.

---

## Dataset

| Atributo | Valor |
|---|---|
| Fonte | IBM Telco Customer Churn (público) |
| Tamanho | 7.043 clientes |
| Features | 19 (após remoção de `customerID` e `gender`) |
| Target | `Churn` (Yes/No → 1/0) |
| Desbalanceamento | ~26,5% churn · ~73,5% não churn |
| Split | 80% treino / 20% teste — estratificado |
| Hash SHA-256 | Registrado por run no MLflow para rastreabilidade |

**Features utilizadas:** `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.

**Features excluídas:**
- `customerID` — identificador direto, sem valor preditivo (LGPD)
- `gender` — irrelevante preditivamente e fonte potencial de viés de grupo

---

## Pré-processamento

- `TotalCharges`: convertido de string para float (11 registros nulos removidos)
- Features numéricas: `StandardScaler`
- Features categóricas: `OneHotEncoder` com `handle_unknown="ignore"`
- Pipeline reprodutível via `sklearn.pipeline.Pipeline` — mesmo objeto registrado e carregado pelo `ChurnModelService`

---

## Métricas de Desempenho

Avaliação no conjunto de teste (hold-out 20%, seed 42).

### Modelos Baseline e Ensembles (Scikit-Learn)

| Modelo | ROC-AUC | F1-Macro | Precisão | Recall |
|---|---|---|---|---|
| DummyClassifier | 0.500 | 0.380 | 0.390 | 0.500 |
| Regressão Logística | **0.836** | **0.740** | 0.748 | 0.736 |
| Random Forest | 0.821 | 0.727 | 0.734 | 0.724 |
| Gradient Boosting | 0.828 | 0.734 | 0.741 | 0.731 |
| XGBoost | 0.826 | 0.731 | 0.738 | 0.728 |
| LightGBM | 0.830 | 0.736 | 0.743 | 0.733 |

### MLP PyTorch

| Configuração | ROC-AUC | F1-Macro |
|---|---|---|
| MLP `[128, 64, 32]` dropout `[0.3, 0.2, 0.0]` | 0.829 | 0.732 |

### Validação Cruzada (5-fold estratificado)

| Métrica | Média | Desvio padrão |
|---|---|---|
| ROC-AUC | 0.845 | ±0.012 |
| F1-Macro | 0.748 | ±0.018 |

### Análise de Custo

| Tipo de Erro | Impacto estimado | Descrição |
|---|---|---|
| Falso Negativo (FN) | R$ 500 por cliente | Cliente churna sem ser detectado |
| Falso Positivo (FP) | R$ 50 por cliente | Ação de retenção desnecessária |

**Razão FN/FP = 10:1** — justifica otimizar recall ajustando `PREDICTION_THRESHOLD` abaixo de 0.5.

---

## Limitações

1. **Dataset estático:** snapshot único, sem garantia de estabilidade temporal.
2. **Contexto geográfico:** dados de empresa fictícia norte-americana; padrões podem diferir no Brasil.
3. **Desbalanceamento de classes:** ~26,5% de churn pode causar subestimação em populações com taxa muito diferente.
4. **Features históricas:** `TotalCharges` e `tenure` dependem de dados históricos do cliente.
5. **Sem monitoramento de drift ativo:** mudanças no comportamento de clientes degradam o modelo sem sinal imediato.

---

## Considerações Éticas e Vieses

### Variáveis excluídas por risco de viés

| Feature | Motivo |
|---|---|
| `gender` | Sem poder preditivo significativo; uso pode perpetuar discriminação |
| `customerID` | Identificador direto — risco de re-identificação (LGPD Art. 5º, I) |

### Riscos conhecidos

- **Feedback loop:** clientes não abordados que cancelam nunca entram como FN nos dados de retreino.
- **Equidade por grupo:** clientes `SeniorCitizen` (~16% do dataset) podem ter performance inferior à média global.
- **`tenure` como proxy:** clientes novos têm maior probabilidade de churn prevista independentemente do comportamento real.

### Conformidade LGPD

- Nenhum PII direto no pipeline de treino ou inferência.
- `customerID` removido antes do pré-processamento.
- Dataset público sem dados sensíveis de clientes reais brasileiros.

---

## Cenários de Falha

| Cenário | Sintoma | Mitigação |
|---|---|---|
| Feature com valor inválido (ex: `tenure = -1`) | Predição silenciosamente incorreta | Validação Pydantic na API rejeita valores `< 0` |
| Modelo não carregado no startup | `HTTPException 500` em `/predict` | `load()` chamado no lifespan; monitorar `/health` |
| MLflow Registry indisponível | API não sobe | Definir `MODEL_URI` para path local como fallback |
| Drift severo em `MonthlyCharges` | Queda de AUC em produção | Alerta quando AUC cai > 5% |
| Novo valor categórico desconhecido | Encoder ignora (`handle_unknown="ignore"`) | Feature fica zerada — monitorar distribuição |

---

## Plano de Monitoramento

| Métrica | Frequência | Alerta |
|---|---|---|
| ROC-AUC em produção | Semanal | < 0.80 |
| Taxa de churn prevista vs. observada | Mensal | Desvio > 5 p.p. |
| Distribuição de `tenure` nas requisições | Diária | KS-test p-value < 0.05 |
| Distribuição de `MonthlyCharges` | Diária | KS-test p-value < 0.05 |
| Latência do endpoint `/predict` (p95) | Contínuo | > 500 ms |
| Taxa de erros HTTP 5xx | Contínuo | > 1% |

### Playbook de resposta

1. **Drift detectado** → comparar histogramas com baseline de treino; se confirmado, iniciar retreino.
2. **Queda de AUC > 5%** → retreinar com dados recentes, registrar nova versão no MLflow, testar em staging antes de promover alias `champion`.
3. **Erro 500 na API** → verificar disponibilidade do MLflow Registry e variável `MLFLOW_TRACKING_URI`.
4. **Retreino:** `make run` executa `ChurnPipeline` e registra nova versão automaticamente.

---

## Informações Técnicas de Deploy

| Componente | Tecnologia | Porta |
|---|---|---|
| API de inferência | FastAPI + Uvicorn | 8000 |
| Servidor MLflow | MLflow 3.x | 5001 |
| Armazenamento de artefatos | MinIO (S3-compatible) | 9000 |
| Backend de metadados | PostgreSQL 15 | 5432 |
| Deploy alternativo | AWS Lambda + Mangum | — |

---

## Histórico de Versões

| Versão | Data | Descrição |
|---|---|---|
| 1.0 | Jun/2026 | Versão inicial — pipeline sklearn + MLP PyTorch, API FastAPI, deploy Docker/Lambda |
