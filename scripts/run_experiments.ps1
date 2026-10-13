# Roda todos os experimentos (treino + avaliação) na GPU e gera a comparação.
# Experimentos já concluídos (com results/<nome>/eval_metrics.json) são pulados, então dá para
# interromper e rodar de novo.
# Uso (na raiz do projeto):  .\scripts\run_experiments.ps1
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
function Run { & $py @args; if ($LASTEXITCODE -ne 0) { throw "Falhou: python $args" } }
function Done($name) { Test-Path "results\$name\eval_metrics.json" }

# 0) Dados
if (-not (Test-Path "data\test.jsonl")) { Run src/prepare_data.py }

# 1) Linhas de base sem fine-tuning (zero-shot)
if (-not (Done "base_qwen1.5b")) { Run src/evaluate.py --run-name base_qwen1.5b }
if (-not (Done "base_qwen0.5b")) { Run src/evaluate.py --run-name base_qwen0.5b --model Qwen/Qwen2.5-0.5B-Instruct }

# 2) Experimentos de fine-tuning: nome, argumentos
$experiments = @(
    @("ft_lr2e-4_r16_ep1", "--lr 2e-4 --lora-r 16 --epochs 1"),   # configuração de referência
    @("ft_lr2e-4_r16_ep1_500ex", "--lr 2e-4 --lora-r 16 --epochs 1 --train-samples 500"),  # menos dados
    @("ft_qwen0.5b_lr2e-4_r16_ep1", "--lr 2e-4 --lora-r 16 --epochs 1 --model Qwen/Qwen2.5-0.5B-Instruct"),
    @("ft_lr5e-5_r16_ep1", "--lr 5e-5 --lora-r 16 --epochs 1"),   # learning rate menor
    @("ft_lr1e-3_r16_ep1", "--lr 1e-3 --lora-r 16 --epochs 1"),   # learning rate maior
    @("ft_lr2e-4_r4_ep1",  "--lr 2e-4 --lora-r 4 --epochs 1"),    # rank LoRA menor
    @("ft_lr2e-4_r64_ep1", "--lr 2e-4 --lora-r 64 --epochs 1"),   # rank LoRA maior
    @("ft_lr2e-4_r16_ep3", "--lr 2e-4 --lora-r 16 --epochs 3")    # mais épocas
)

foreach ($e in $experiments) {
    $name = $e[0]
    if (Done $name) { Write-Host "pulando $name (já concluído)"; continue }
    $extra = $e[1] -split " "
    Write-Host "`n===== $name =====" -ForegroundColor Cyan
    Run src/train.py --run-name $name @extra
    $modelArg = @()
    if ($e[1] -match "--model (\S+)") { $modelArg = @("--model", $Matches[1]) }
    Run src/evaluate.py --run-name $name --adapter "outputs/$name" @modelArg
}

# 3) Efeito da temperatura na inferência (configuração de referência)
if (-not (Done "ft_lr2e-4_r16_ep1_temp1.0")) {
    Run src/evaluate.py --run-name ft_lr2e-4_r16_ep1_temp1.0 --adapter outputs/ft_lr2e-4_r16_ep1 --temperature 1.0
}

# 4) Tabela, gráficos e significância estatística
Run src/compare_results.py
Run src/significance.py

# 5) Exporta o adaptador recomendado para models/ (versionado, usado pelo frontend)
Run src/export_adapter.py --run-name ft_lr2e-4_r16_ep1
Write-Host "`nEXPERIMENTOS CONCLUIDOS" -ForegroundColor Green
