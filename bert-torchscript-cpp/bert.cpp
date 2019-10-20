#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  } 
  
  auto methods = module.get_methods();
  std::cout << "Methods" << std::endl;
  for(const torch::jit::script::Method method : methods){
    std::cout << "\t"+method.name() << std::endl;    
  }


  // Run Forward
  std::vector<torch::jit::IValue> inputs;  
  inputs.push_back(torch::ones({1, 1}, torch::dtype(torch::kLong)));
  inputs.push_back(torch::zeros({1, 1}, torch::dtype(torch::kLong)));

  

  auto output = module.forward(inputs).toTuple();
  
  auto tensor = output->elements()[0].toTensor();
  std::cout << tensor << '\n';

  std::cout << "Done\n";
}