#!/bin/bash

# Cores para o terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

MODEL="assets/STLN_modelv2.onnx"

echo -e "${BLUE}Configurando Microfone Virtual AMD...${NC}"

# 1. Criar o sink nulo (para onde o C++ vai enviar o áudio)
SINK_ID=$(pactl load-module module-null-sink \
    sink_name=AMD_Noise_Suppression_Sink \
    sink_properties=device.description="AMD_Noise_Suppression_Internal")

if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao criar Sink Virtual. Verifique se o PulseAudio/PipeWire está rodando.${NC}"
    exit 1
fi

# 2. Criar a fonte virtual (o que aparece como Microfone selecionável)
SOURCE_ID=$(pactl load-module module-virtual-source \
    source_name=AMD_Noise_Suppression \
    master=AMD_Noise_Suppression_Sink.monitor \
    source_properties=device.description="AMD\ Noise\ Suppression")

echo -e "${GREEN}✓ Dispositivo 'AMD Noise Suppression' criado no sistema!${NC}"
echo -e "${BLUE}Iniciando processador OpenVINO...${NC}"

# 3. Executar o processador C++ direcionando a saída para o nosso Sink
# Usamos --output "AMD_Noise_Suppression_Sink" para garantir que o PortAudio mande para lá
./amd_noise_suppression "$MODEL" --output "AMD_Noise_Suppression_Sink" "$@"

# 4. Limpeza ao fechar
echo -e "\n${BLUE}Removendo dispositivos virtuais...${NC}"
pactl unload-module "$SOURCE_ID"
pactl unload-module "$SINK_ID"
echo -e "${GREEN}Finalizado.${NC}"
