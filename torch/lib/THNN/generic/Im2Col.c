#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Im2Col.c"
#else

void THNN_(Im2Col_updateOutput)(
        THNNState *state,
        THTensor *input,
        THTensor *output,
        int kW, int kH,
        int dW, int dH,
        int padW, int padH,
        int sW, int sH)
{
    THAssertMsg(false, "Not implemented for CPU");
}

void THNN_(Im2Col_updateGradInput)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *gradInput,
        int kW, int kH,
        int dW, int dH,
        int padW, int padH,
        int sW, int sH)
{
    THAssertMsg(false, "Not implemented for CPU");
}

#endif
