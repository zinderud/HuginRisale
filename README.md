# HuginRisale

https://huggingface.co/zinderud/hugin-risale-gemma-3-1b/tree/main

sadece belirli sosyaları indirmek için
pip install huggingface_hub

huggingface-cli download zinderud/risale-sohbet-turkish --include "transcripts/*" --local-dir ./content/risale-sohbet
huggingface-cli download --repo-type dataset zinderud/risale-sohbet-turkish --include "transcripts/*" --local-dir ./content/risale-sohbet