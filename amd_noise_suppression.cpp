#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <openvino/openvino.hpp>
#include <portaudio.h>
#include <fftw3.h>

#define SAMPLE_RATE 48000
#define FFT_SIZE 512
#define HOP_SIZE 256
#define BINS 257

class AMDNoiseSuppression {
public:
    AMDNoiseSuppression(const std::string& model_path, const std::string& device_name = "GPU") {
        
        // Initialize OpenVINO Core
        ov::Core core;

        // Debug: List available devices
        std::cout << "Dispositivos OpenVINO disponíveis:" << std::endl;
        for (auto&& dev : core.get_available_devices()) {
            std::cout << " - " << dev << std::endl;
        }
        
        // Load the model
        auto model = core.read_model(model_path);
        
        // Configure for GPU/CPU
        try {
            compiled_model = core.compile_model(model, device_name);
            current_device = device_name;
        } catch (const std::exception& e) {
            std::cerr << "Aviso: Falha ao carregar na " << device_name << ": " << e.what() << std::endl;
            std::cerr << "Tentando carregar na CPU..." << std::endl;
            compiled_model = core.compile_model(model, "CPU");
            current_device = "CPU";
        }

        infer_request = compiled_model.create_infer_request();

        // Get input/output information
        input_mag = infer_request.get_input_tensor(0); // input_2
        input_state = infer_request.get_input_tensor(1); // input_3

        // Setup shape for inputs
        // input_state is [1, 3, 280, 2]
        float* state_ptr = input_state.data<float>();
        std::fill(state_ptr, state_ptr + input_state.get_size(), 0.0f);

        // FFT config
        fft_input = (double*) fftw_malloc(sizeof(double) * FFT_SIZE);
        fft_output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BINS);
        ifft_input = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BINS);
        ifft_output = (double*) fftw_malloc(sizeof(double) * FFT_SIZE);

        plan_forward = fftw_plan_dft_r2c_1d(FFT_SIZE, fft_input, fft_output, FFTW_MEASURE);
        plan_backward = fftw_plan_dft_c2r_1d(FFT_SIZE, ifft_input, ifft_output, FFTW_MEASURE);

        // Window and buffers
        in_buffer.assign(FFT_SIZE, 0.0f);
        out_buffer.assign(FFT_SIZE, 0.0f);
        window.resize(FFT_SIZE);
        for(int i=0; i<FFT_SIZE; ++i) {
            float hann = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FFT_SIZE - 1)));
            window[i] = sqrtf(hann);
        }
    }

    ~AMDNoiseSuppression() {
        fftw_destroy_plan(plan_forward);
        fftw_destroy_plan(plan_backward);
        fftw_free(fft_input);
        fftw_free(fft_output);
        fftw_free(ifft_input);
        fftw_free(ifft_output);
    }

    void process(const float* in, float* out) {
        // Step 1: Buffer and Windowing
        std::copy(in_buffer.begin() + HOP_SIZE, in_buffer.end(), in_buffer.begin());
        std::copy(in, in + HOP_SIZE, in_buffer.end() - HOP_SIZE);

        for(int i=0; i<FFT_SIZE; ++i) fft_input[i] = (double)(in_buffer[i] * window[i]);

        // Step 2: FFT
        fftw_execute(plan_forward);

        // Step 3: Magnitude calculation into OpenVINO tensor
        float* mag_ptr = input_mag.data<float>();
        for(int i=0; i<BINS; ++i) {
            mag_ptr[i] = sqrtf((float)(fft_output[i][0]*fft_output[i][0] + fft_output[i][1]*fft_output[i][1]));
        }

        // Step 4: OpenVINO Inference
        // Note: input_state is already set in the infer_request (initially zeros)
        infer_request.infer();
        
        // Get outputs
        auto output_mask_tensor = infer_request.get_output_tensor(0); // tf.math.sigmoid_12
        auto output_state_tensor = infer_request.get_output_tensor(1); // tf.stack_2
        
        const float* mask = output_mask_tensor.data<const float>();
        
        // Update persistent state for next frame
        std::copy(output_state_tensor.data<const float>(), 
                  output_state_tensor.data<const float>() + output_state_tensor.get_size(), 
                  input_state.data<float>());

        // Step 5: Mask application and IFFT
        for(int i=0; i<BINS; ++i) {
            ifft_input[i][0] = fft_output[i][0] * mask[i];
            ifft_input[i][1] = fft_output[i][1] * mask[i];
        }
        fftw_execute(plan_backward);

        // Step 6: Overlap-Add
        std::copy(out_buffer.begin() + HOP_SIZE, out_buffer.end(), out_buffer.begin());
        std::fill(out_buffer.end() - HOP_SIZE, out_buffer.end(), 0.0f);
        for(int i=0; i<FFT_SIZE; ++i) {
            out_buffer[i] += (float)((ifft_output[i] / FFT_SIZE) * window[i]);
        }

        if (out) {
            std::copy(out_buffer.begin(), out_buffer.begin() + HOP_SIZE, out);
        }
    }

    std::string get_active_device() const { return current_device; }

    static void list_devices() {
        Pa_Initialize();
        int numDevices = Pa_GetDeviceCount();
        std::cout << "Dispositivos de Áudio Disponíveis:" << std::endl;
        for (int i = 0; i < numDevices; i++) {
            const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
            std::cout << i << ": " << deviceInfo->name 
                      << " (In: " << deviceInfo->maxInputChannels 
                      << ", Out: " << deviceInfo->maxOutputChannels << ")" << std::endl;
        }
        Pa_Terminate();
    }

    static int find_device_by_name(const std::string& name, bool is_input) {
        int numDevices = Pa_GetDeviceCount();
        for (int i = 0; i < numDevices; i++) {
            const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
            std::string dname = deviceInfo->name;
            if (dname.find(name) != std::string::npos) {
                if (is_input && deviceInfo->maxInputChannels > 0) return i;
                if (!is_input && deviceInfo->maxOutputChannels > 0) return i;
            }
        }
        return -1;
    }

private:
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor input_mag, input_state;
    std::string current_device;

    double *fft_input, *ifft_output;
    fftw_complex *fft_output, *ifft_input;
    fftw_plan plan_forward, plan_backward;

    std::vector<float> in_buffer, out_buffer, window;
};

static int paCallback(const void *inputBuffer, void *outputBuffer,
                       unsigned long framesPerBuffer,
                       const PaStreamCallbackTimeInfo* timeInfo,
                       PaStreamCallbackFlags statusFlags,
                       void *userData) {
    if (!inputBuffer) return paContinue;
    auto* suppressor = (AMDNoiseSuppression*)userData;
    suppressor->process((const float*)inputBuffer, (float*)outputBuffer);
    return paContinue;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <caminho_do_modelo.onnx> [opções]" << std::endl;
        std::cerr << "Opções:" << std::endl;
        std::cerr << "  --monitor       Habilita eco p/ teste" << std::endl;
        std::cerr << "  --cpu           Força uso de CPU" << std::endl;
        std::cerr << "  --list          Lista dispositivos" << std::endl;
        std::cerr << "  --input <idx/nome>  Seleciona entrada (Microfone Real)" << std::endl;
        std::cerr << "  --output <idx/nome> Seleciona saída (Monitor/Virtual Sink)" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    bool monitor = false;
    std::string device = "GPU";
    int input_idx = -1;
    int output_idx = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--list") {
            AMDNoiseSuppression::list_devices();
            return 0;
        }
        if (arg == "--monitor") monitor = true;
        if (arg == "--cpu") device = "CPU";
        if (arg == "--input" && i+1 < argc) {
            std::string val = argv[++i];
            try { input_idx = std::stoi(val); } 
            catch (...) { 
                Pa_Initialize();
                input_idx = AMDNoiseSuppression::find_device_by_name(val, true); 
                Pa_Terminate();
            }
        }
        if (arg == "--output" && i+1 < argc) {
            std::string val = argv[++i];
            try { output_idx = std::stoi(val); } 
            catch (...) { 
                Pa_Initialize();
                output_idx = AMDNoiseSuppression::find_device_by_name(val, false);
                Pa_Terminate();
            }
        }
    }

    try {
        AMDNoiseSuppression suppressor(model_path, device);

        Pa_Initialize();
        
        if (input_idx == -1) input_idx = Pa_GetDefaultInputDevice();
        if (output_idx == -1) {
            if (monitor) output_idx = Pa_GetDefaultOutputDevice();
            else output_idx = -1; // No output if not monitoring and no virtual device specified
        }

        PaStreamParameters inputParams, outputParams;
        inputParams.device = input_idx;
        inputParams.channelCount = 1;
        inputParams.sampleFormat = paFloat32;
        inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
        inputParams.hostApiSpecificStreamInfo = nullptr;

        PaStreamParameters* pOutputParams = nullptr;
        if (output_idx != -1) {
            outputParams.device = output_idx;
            outputParams.channelCount = 1;
            outputParams.sampleFormat = paFloat32;
            outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
            outputParams.hostApiSpecificStreamInfo = nullptr;
            pOutputParams = &outputParams;
        }

        PaStream *stream;
        Pa_OpenStream(&stream, &inputParams, pOutputParams, SAMPLE_RATE, HOP_SIZE, paClipOff, paCallback, &suppressor);
        Pa_StartStream(stream);

        std::cout << "\nAMD Noise Suppression (OpenVINO) Ativo" << std::endl;
        std::cout << "---------------------------------------" << std::endl;
        std::cout << "Hardware: " << suppressor.get_active_device() << std::endl;
        std::cout << "Entrada: [" << input_idx << "] " << Pa_GetDeviceInfo(input_idx)->name << std::endl;
        if (output_idx != -1)
            std::cout << "Saída:   [" << output_idx << "] " << Pa_GetDeviceInfo(output_idx)->name << std::endl;
        else
            std::cout << "Saída:   Desativada (Modo Silencioso)" << std::endl;
        
        std::cout << "Pressione Enter para parar." << std::endl;
        std::cin.get();

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
    } catch (const std::exception& e) {
        std::cerr << "Erro fatal: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
