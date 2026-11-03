# Registro histórico de execução Azure

Preservamos o relatório de áudio produzido antes da revisão do cruzamento
entre entidades clínicas e léxico. O documento registra os provedores
Azure Speech to Text e Text Analytics, a transcrição e WER de 4%.

A lista de termos contém expansões indevidas, como “dor insuportável”,
que não aparecem na transcrição. Portanto, este arquivo é evidência de uso
dos serviços, não uma saída corrigida nem uma referência clínica.

A implementação atual exige suporte textual para cada termo. Os testes
automatizados verificam essa regressão com respostas controladas do SDK.
A execução corrigida foi concluída com Speech e Language reais em 3 de
novembro de 2026 e está em `../azure_real/`. Este diretório permanece
como registro histórico e não deve ser apresentado como resultado atual.
