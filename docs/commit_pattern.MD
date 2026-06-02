# Padrão de commits (Conventional Commits)

Este documento descreve um padrão sugerido de histórico de commits baseado em *Conventional Commits*. Use-o para manter um histórico consistente e facilitar geração de changelogs, versionamento semântico e automações.

## Formato básico

Um commit deve seguir o formato:

```
<tipo>(<escopo opcional>): <descrição curta>

[corpo opcional]

[rodapé opcional]
```

Tipos comuns:

- `feat`: nova funcionalidade
- `fix`: correção de bug
- `docs`: mudanças na documentação
- `chore`: tarefas de manutenção (config, build, etc.)
- `test`: adicionar/ajustar testes
- `refactor`: refatoração sem mudança de comportamento
- `perf`: melhoria de performance
- `ci`: alterações em CI/CD

Exemplo com escopo:

```
feat(api): add predict endpoint
```

## Exemplos sugeridos (histórico)

Use exemplos como referência ao escrever commits:

- `chore: initial repo structure and pyproject.toml`
- `chore: add Makefile, .gitignore and Dockerfile`
- `docs: add architecture.md and ML Canvas`
- `feat: add EDA notebook with data quality report`
- `feat(data): add pandera schema and preprocessing pipeline`
- `feat(models): add baseline models with MLflow tracking`
- `feat(models): implement ChurnMLP with BatchNorm and Dropout`
- `feat(train): add training loop with early stopping and pos_weight`
- `feat(api): add FastAPI predict and health endpoints`
- `test: add smoke, schema and api tests`
- `docs: add model card and monitoring plan`
- `fix(api): fix ruff lint errors in src/api/predictor.py`

## Boas práticas

- Seja imperativo e no tempo presente na descrição curta (`add`, `fix`, `update`).
- Mantenha a linha de descrição curta com até ~72 caracteres.
- Use o corpo do commit para explicar o *porquê* da mudança quando não óbvio.
- Inclua referências a issues no rodapé (`Closes #123`).

## Modelo rápido

```
<tipo>(<escopo>): <descrição curta>

Descrição mais detalhada (opcional), justificativa e notas de implementação.

Closes: #<issue>
```

---
