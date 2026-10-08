param([Parameter(ValueFromRemainingArguments=$true)][string[]]$TrainingArgs)
python (Join-Path $PSScriptRoot 'src/train.py') @TrainingArgs
exit $LASTEXITCODE
