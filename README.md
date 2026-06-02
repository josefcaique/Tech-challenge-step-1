🐍 Ambiente Python com pipx + Poetry
Guia introdutório para configurar um ambiente Python moderno usando pipx e Poetry.
---
📋 Pré-requisitos
Python 3.8 ou superior instalado
`pip` disponível no terminal
Verifique sua versão do Python:
```bash
python --version
# ou
python3 --version
```
---
1️⃣ Instalando o pipx
O pipx instala ferramentas de linha de comando Python em ambientes isolados, evitando conflitos de dependências.
Linux / macOS
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```
Após o comando `ensurepath`, feche e abra o terminal novamente para atualizar o `PATH`.
Windows
```powershell
python -m pip install --user pipx
python -m pipx ensurepath
```
> Reinicie o terminal após a instalação.
Verificando a instalação
```bash
pipx --version
```
---
2️⃣ Instalando o Poetry com pipx
O Poetry é um gerenciador de dependências e empacotador moderno para Python.
```bash
pipx install poetry
```
Verificando a instalação
```bash
poetry --version
```
Atualizando o Poetry futuramente
```bash
pipx upgrade poetry
```

Entrar no projeto:
```bash
cd meu-projeto
poetry init
```
---
4️⃣ Entendendo o `pyproject.toml`
O `pyproject.toml` é o arquivo central de configuração do projeto. Ele substitui o antigo `setup.py` e `requirements.txt`.
```toml
[tool.poetry]
name = "meu-projeto"
version = "0.1.0"
description = "Descrição do meu projeto"
authors = ["Seu Nome <seu@email.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```
Seção	O que faz
`[tool.poetry]`	Metadados do projeto (nome, versão, autor)
`[tool.poetry.dependencies]`	Dependências de produção
`[tool.poetry.group.dev.dependencies]`	Dependências apenas para desenvolvimento
`[build-system]`	Define como o projeto é empacotado
---
5️⃣ Instalando dependências
Adicionar uma dependência de produção
```bash
poetry add requests
poetry add fastapi
```
Adicionar uma dependência de desenvolvimento
```bash
poetry add --group dev pytest
poetry add --group dev ruff
```
Instalar todas as dependências do projeto
```bash
poetry install
```
Instalar apenas dependências de produção (sem dev)
```bash
poetry install --without dev
```
---
6️⃣ Ativando o ambiente virtual com `poetry shell`
O Poetry cria e gerencia um ambiente virtual isolado para cada projeto.
```bash
poetry shell
```
Isso ativa o virtualenv do projeto. Você verá o nome do ambiente no prompt:
```
(meu-projeto-py3.11) $
```
Rodando um comando sem ativar o shell
```bash
poetry run python meu_script.py
poetry run pytest
poetry run mlflow --version
poetry run mlflow ui
```
Se o terminal disser que `mlflow` não foi encontrado, execute `poetry shell` antes do comando ou use sempre `poetry run mlflow ...`. No Poetry, o executável não fica disponível como comando global do PowerShell. Para subir a interface, prefira `poetry run mlflow ui` ou `make mlflow-ui`.
Saindo do ambiente virtual
```bash
exit
```
---
7️⃣ Fluxo completo — do zero ao ambiente rodando
```bash
# 1. Instalar pipx (uma vez só na máquina)
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# 2. Instalar Poetry via pipx (uma vez só na máquina)
pipx install poetry


cd tech-challenge

# 4. Adicionar dependências
poetry add requests

# 5. Ativar o ambiente
poetry shell

```
---
🔍 Comandos úteis do Poetry
Comando	Descrição
`poetry add <pacote>`	Adiciona uma dependência
`poetry remove <pacote>`	Remove uma dependência
`poetry install`	Instala todas as dependências
`poetry update`	Atualiza as dependências
`poetry show`	Lista os pacotes instalados
`poetry shell`	Ativa o ambiente virtual
`poetry run <cmd>`	Roda um comando no ambiente
`poetry env info`	Mostra info do virtualenv
`poetry build`	Empacota o projeto
`poetry publish`	Publica no PyPI
---
📚 Referências
Documentação oficial do pipx
Documentação oficial do Poetry
Especificação do pyproject.toml (PEP 518)