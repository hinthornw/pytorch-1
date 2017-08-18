#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Im2Col.cu"
#else

static inline void THNN_(Im2Col_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int kH, int kW, int dH, int dW,
                         int padH, int padW, int sH, int sW) {
  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(sW > 0 && sH > 0, 11,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  THCUNN_argCheck(state, ndim == 3, 2, input,
                  "3D input tensor expected but got: %s");

  long nInputPlane  = input->size[dimf];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long outputHeight = (inputHeight + 2*padH - kH - ((kH - 1)*(dH - 1))) / sH + 1;
  long outputWidth  = (inputWidth + 2*padW - kW - ((kW - 1)*(dW - 1))) / sW + 1;
  long nOutputPlane = nInputPlane * kW * kH;
  long outputLength = outputHeight * outputWidth;

  if (outputWidth < 1 || outputHeight < 1)
      THError("Given input size: (%d x %d x %d). "
              "Calculated output size: (%d x %d). Output size is too small",
              nInputPlane,inputHeight,inputWidth,nOutputPlane,outputLength);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, 0, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim, 1, outputLength);
  }
}

void THNN_(Im2Col_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int sW, int sH) {

  THCUNN_assertSameGPU(state, 2, input, output);

  // Params:
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long nInputPlane = input->size[dimf];
  long outputHeight = (inputHeight + 2*padH - (dH * (kH - 1)) + 1) / sH + 1;
  long outputWidth  = (inputWidth + 2*padW - (dW * (kW - 1)) + 1) / sW + 1;
  long nOutputPlane = nInputPlane * kW * kH;
  long outputLength = outputHeight*outputWidth;

  THNN_(Im2Col_shapeCheck)
       (state, input, NULL, kH, kW, dH, dW, padH, padW, sH, sW);

  input = THCTensor_(newContiguous)(state, input);

  // Resize output
  THCTensor_(resize2d)(state, output, nOutputPlane, outputLength);

  // Extract columns:
  im2col(
    THCState_getCurrentStream(state),
    THCTensor_(data)(state, input),
    nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, sH, sW,
    dH, dW, THCTensor_(data)(state, output)
  );

  THCTensor_(free)(state, input);
}

void THNN_(Im2Col_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int sW, int sH) {

  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  // Params
  long inputHeight  = input->size[1];
  long inputWidth   = input->size[2];
  long nInputPlane = input->size[0];

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  THNN_(Im2Col_shapeCheck)
       (state, input, gradOutput, kH, kW, dH, dW, padH, padW, sH, sW);

  // Resize output
  THCTensor_(resize3d)(state, gradInput, nInputPlane, inputHeight, inputWidth);

  // Unpack columns back into input:
  col2im<real, accreal>(
    THCState_getCurrentStream(state),
    THCTensor_(data)(state, gradOutput),
    nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, sH, sW,
    dH, dW, THCTensor_(data)(state, gradInput)
  );

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}


#endif
