#include "NeuralNetwork.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "person_detect_model_data.h"
#include "model_settings.h"
#include "detection_responder.h"

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kArenaSize = 96 * 1024;

NeuralNetwork::NeuralNetwork()
{
  error_reporter = new tflite::MicroErrorReporter();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  micro_op_resolver = new tflite::MicroMutableOpResolver<3>();
  micro_op_resolver->AddDepthwiseConv2D();
  micro_op_resolver->AddConv2D();
  micro_op_resolver->AddAveragePool2D();

  tensor_arena = (uint8_t *)malloc(kArenaSize);
   if (!tensor_arena)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(
        model, *micro_op_resolver, tensor_arena, kArenaSize, error_reporter);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

uint8_t *NeuralNetwork::getInputBuffer()
{
    return input->data.uint8;
}



void NeuralNetwork::detect()
{
    // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  uint8_t person_score = output->data.uint8[kPersonIndex];
  uint8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  RespondToDetection(error_reporter, person_score, no_person_score);
}
