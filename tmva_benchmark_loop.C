#include "TROOT.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/IMethod.h"
#include "TMVA/MethodBase.h"
#include "TFile.h"

#include <iostream>
#include <fstream>
#include <vector>


void benchmark()
{
    std::ofstream file;
    file.open("benchmark_results.csv");

    TFile* outputFile = TFile::Open("Higgs_ClassificationOutput.root", "RECREATE");
    TFile* dataFile = TFile::Open("Higgs_data.root");

    TMVA::DataLoader* dataloader = new TMVA::DataLoader("dataset");

    dataloader->AddVariable("m_jj");
    dataloader->AddVariable("m_jjj");
    dataloader->AddVariable("m_lv");
    dataloader->AddVariable("m_jlv");
    dataloader->AddVariable("m_bb");
    dataloader->AddVariable("m_wbb");
    dataloader->AddVariable("m_wwbb");

    dataloader->AddVariable("log(m_jj)");
    dataloader->AddVariable("log(m_jjj)");
    dataloader->AddVariable("log(m_lv)");
    dataloader->AddVariable("log(m_jlv)");
    dataloader->AddVariable("log(m_bb)");
    dataloader->AddVariable("log(m_wbb)");
    dataloader->AddVariable("log(m_wwbb)");

    dataloader->AddVariable("sqrt(m_jj)");
    dataloader->AddVariable("sqrt(m_jjj)");
    dataloader->AddVariable("sqrt(m_lv)");
    dataloader->AddVariable("sqrt(m_jlv)");
    dataloader->AddVariable("sqrt(m_bb)");
    dataloader->AddVariable("sqrt(m_wbb)");
    dataloader->AddVariable("sqrt(m_wwbb)");

    dataloader->AddVariable("2^m_jj");
    dataloader->AddVariable("2^m_jjj");
    dataloader->AddVariable("2^m_lv");
    dataloader->AddVariable("2^m_jlv");

    TTree *signalTree     = (TTree*)dataFile->Get("sig_tree");
    TTree *backgroundTree = (TTree*)dataFile->Get("bkg_tree");

    Double_t signalWeight     = 1.0;
    Double_t backgroundWeight = 1.0;

    dataloader->AddSignalTree    ( signalTree,     signalWeight     );
    dataloader->AddBackgroundTree( backgroundTree, backgroundWeight );


    TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
    TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

    dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
                                    "nTrain_Signal=0:nTrain_Background=0"
                                    ":SplitMode=Random:NormMode=NumEvents:!V" );

    std::vector<TString> methodNames;

    TMVA::Factory factory("TMVA_Higgs_Classification", outputFile,
                              "!V:ROC:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" );

    for (auto num_layers : {1, 2}) {
        for (auto layer_size : {15, 30, 130}) {
            for (auto batch_size : {1, 8}) {
            
                TString layerConfigString = ",RELU|" + std::to_string(layer_size);
            
                TString layoutString ("Layout=RELU|" + std::to_string(layer_size));
                for (int i = 1; i < num_layers; ++i) {
                    layoutString += layerConfigString;
                }
                layoutString += ",SOFTMAX";

                TString training1("LearningRate=1e-1,Momentum=0.9,Repetitions=1,"
                            "ConvergenceSteps=1,BatchSize=" + std::to_string(batch_size)
                             + ",TestRepetitions=5,MaxEpochs=1,"
                            "WeightDecay=1e-4,Regularization=L2,"
                            "DropConfig=0.0+0.2+0.2+0.");

                TString trainingStrategyString ("TrainingStrategy=");
                trainingStrategyString += training1;

                TString dnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"
                              "WeightInitialization=XAVIERUNIFORM");
                dnnOptions.Append (":"); dnnOptions.Append (layoutString);
                dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString);

                dnnOptions += ":Architecture=CPU";
                std::cout << "All good here" << dnnOptions << std::endl;
                TString methodName = "DNN_CPU_RUN" + TString("") +
                                 "_BatchSize" + std::to_string(batch_size) +
                                 "_Depth" + std::to_string(num_layers) +
                                 "_Width" + std::to_string(layer_size);

                factory.BookMethod(dataloader, TMVA::Types::kDNN, methodName, dnnOptions);
               
                methodNames.push_back(methodName);
            }
        }
    }

    for (TString methodName : methodNames) {
        TMVA::IMethod * imethod = factory.GetMethod("dataset", methodName);
        TMVA::MethodBase *method = dynamic_cast<TMVA::MethodBase *>(imethod);

        method->AddOutput(TMVA::Types::kTesting, TMVA::Types::kClassification);
        file << "TMVA_1," << method->GetName() << ","
        << method->GetTestTime() / 10000. << std::endl;
    }
    file.close();
}

/* DOESN'T WORK. REQUIRES RECOMPILATION. `GetMvaValues` is a protected member of `MethodBase` 
for (TString methodName : methodNames) {
    TMVA::IMethod * imethod = factory.GetMethod("dataset", methodName);
    TMVA::MethodBase *method = dynamic_cast<TMVA::MethodBase *>(imethod);

    UInt_t firstEvt = 0;
    UInt_t lastEvt = 32;
    Bool_t logProgress = false;
    method->GetMvaValues(firstEvt, lastEvt, logProgress);
}
*/
int main()
{
  benchmark();

  return 0;
}

