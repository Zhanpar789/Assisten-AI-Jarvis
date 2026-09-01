#include "whisper.h"

#include <cstring>
#include <new>
#include <string>

struct jarvis_whisper_context {
    whisper_context * whisper;
    int n_threads;
};

extern "C" jarvis_whisper_context * jarvis_whisper_create(
        const char * model_path,
        int n_threads) {
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = true;
    params.flash_attn = true;

    whisper_context * whisper = whisper_init_from_file_with_params(model_path, params);
    if (whisper == nullptr) {
        return nullptr;
    }

    jarvis_whisper_context * context = new (std::nothrow) jarvis_whisper_context{
        whisper,
        n_threads,
    };
    if (context == nullptr) {
        whisper_free(whisper);
    }
    return context;
}

extern "C" void jarvis_whisper_destroy(jarvis_whisper_context * context) {
    if (context == nullptr) {
        return;
    }
    whisper_free(context->whisper);
    delete context;
}

extern "C" int jarvis_whisper_transcribe(
        jarvis_whisper_context * context,
        const float * samples,
        int n_samples,
        char * output,
        int output_size) {
    if (context == nullptr || samples == nullptr || n_samples <= 0 ||
            output == nullptr || output_size <= 0) {
        return -1;
    }
    output[0] = '\0';

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = context->n_threads;
    params.language = "en";
    params.no_context = true;
    params.no_timestamps = true;
    params.single_segment = true;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.suppress_blank = true;
    params.suppress_nst = true;
    params.temperature = 0.0f;
    params.temperature_inc = 0.0f;

    if (whisper_full(context->whisper, params, samples, n_samples) != 0) {
        return -2;
    }

    std::string text;
    const int n_segments = whisper_full_n_segments(context->whisper);
    for (int i = 0; i < n_segments; ++i) {
        const char * segment = whisper_full_get_segment_text(context->whisper, i);
        if (segment != nullptr) {
            text += segment;
        }
    }
    if (text.size() + 1 > static_cast<size_t>(output_size)) {
        return -3;
    }

    std::memcpy(output, text.c_str(), text.size() + 1);
    return 0;
}
