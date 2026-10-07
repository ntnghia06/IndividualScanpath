param([Parameter(ValueFromRemainingArguments=$true)][string[]]$EvaluationArgs)
python (Join-Path $PSScriptRoot 'src/test.py') @EvaluationArgs
exit $LASTEXITCODE
