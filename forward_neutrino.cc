// Headers and Namespaces.

#include "Pythia8/Pythia.h" // Include Pythia headers.
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <list>
#include <vector>
#include <cstdlib>
#include <math.h> 
#include <bits/stdc++.h>
#include <ctime>

using namespace Pythia8;

int main(int argc, char* argv[]) {
    // Set up generation.
    Pythia pythia;
    pythia.readString("130:mayDecay = True");
    pythia.readString("310:mayDecay = True");
    pythia.readString("321:mayDecay = True");
    pythia.readString("411:mayDecay = True");
    pythia.readString("431:mayDecay = True");
    pythia.readString("421:mayDecay = True");
    pythia.readString("211:mayDecay = True");
    pythia.readString("3122:mayDecay = True");
    pythia.readString("3222:mayDecay = True");
    pythia.readString("3112:mayDecay = True");
    pythia.readString("3322:mayDecay = True");
    pythia.readString("3312:mayDecay = True");
    pythia.readString("4122:mayDecay = True");
    
    pythia.readString("Main:timesAllowErrors = 3"); // how many aborts before run stops
    pythia.readString("Init:showChangedSettings = on");

    // Randomizer 
    std::string filename = argv[1];
    int len = filename.length();
    int s1 = stoi(filename.substr(1,len)); 
    int seed = ((time(0)*s1)%900000000)+1;
    pythia.readString("Random:setSeed=on");
    pythia.readString("Random:seed="+std::to_string(seed));

    // Process and kinematic setup
    pythia.readString("Beams:idA = 2212"); 
    pythia.readString("Beams:idB = 2212");   
    pythia.readString("Beams:eCM = 13000.");   
    pythia.readString("SoftQCD:nonDiffractive = on");

    // PDF selection from LHAPDF
    std::string pdfSet = "NNPDF31_nnlo_as_0118";
    pythia.readString("PDF:pSet = LHAPDF6:"+pdfSet);
    pythia.init(); 
    
    //  Event number and parameters
    int nEvent = 10000000;
    double Emin = 10.;
    double px;
    double py;
    double pz;
    double energy;
    double posx;
    double x;
    double posy;
    double y;
    double z;
    double eta;
    double x1;
    double x2;
    int id1;
    int id2;
    double Q;
    int imother;
    int idaughter;
    int mother_ID;
    int daughter_ID;
    int maxiter;
    int part_stat;

    // Detector dimensions
    std::vector<float> _dimF = {-0.125, 0.125,-0.125, 0.125};
    std::vector<float> _dimS = { 0.080, 0.470, 0.155, 0.545};

    ofstream datafile;
    datafile.open("/data/theorie/josepsolac/neutrino_fluxes/data_files/data_"+pdfSet+"/neutrinos"+filename+"_"+pdfSet+".dat");// Start event loop (for nEvent iterations)

    for (int iEvent=1; iEvent <= nEvent; ++iEvent) {

        pythia.next(); // Generate an(other) event. Fill event record.

//      Go through every particle in each event record
        for (int i = 0; i < pythia.event.size(); ++i) {

//          Only keep primary hadrons and run over daughter list, to find a neutrino
            part_stat = pythia.event[i].statusAbs();
            if (part_stat >= 81 and part_stat <= 89) {
                
                vector <int> daughter_list=pythia.event[i].daughterListRecursive();
                maxiter = daughter_list.size();
                
                if (maxiter != 0) {
                    
                    for (int j = 0; j < maxiter; ++j) {

//                      Filter according to pipe conditions and kinematics
                        idaughter = daughter_list[j];
                        pz = pythia.event[idaughter].pz();
                        energy = pythia.event[idaughter].e();
                        
                        if ((pythia.event[idaughter].isLepton()) && (pythia.event[idaughter].isNeutral()) && (energy > Emin) && (pz > 0.)){
                            
                            imother = pythia.event[idaughter].mother1();
                            z = pythia.event[idaughter].zProd()/1000.;
                            
                            if ((pythia.event[imother].isNeutral() && z < 150.) || (pythia.event[imother].isCharged() && z < 22.)) {
                                
                                eta = pythia.event[idaughter].eta();
                                x1 = pythia.info.x1pdf();
                                x2 = pythia.info.x2pdf();
                                id1 = pythia.info.id1pdf();
                                id2 = pythia.info.id2pdf();
                                Q = pythia.info.QRen();
                                mother_ID = pythia.event[i].id();
                                daughter_ID = pythia.event[idaughter].id();

                                px = pythia.event[idaughter].px();
                                py = pythia.event[idaughter].py();
                        
                                x = pythia.event[idaughter].xProd()/1000.;
                                y = pythia.event[idaughter].yProd()/1000.;

                                posx = x + (px/pz)*(480.-z);
                                posy = y + (py/pz)*(480.-z);

                                bool passF = (posx>_dimF[0] and posx<_dimF[1] and posy>_dimF[2] and posy<_dimF[3]);
                                bool passS = (posx>_dimS[0] and posx<_dimS[1] and posy>_dimS[2] and posy<_dimS[3]);
 
                                if (!passF && !passS) continue;
                                if (passF && !passS) {
                                    datafile << 1 << " " << energy << " " << pythia.info.sigmaGen()/nEvent << " " << eta << " " << daughter_ID << " " << mother_ID << " " << id1 << " " << x1 << " " << id2 << " " << x2 << " " << Q << endl;
                                }
                                else if (!passF && passS) {
                                    datafile << 2 << " " << energy << " " << pythia.info.sigmaGen()/nEvent << " " << eta << " " << daughter_ID << " " << mother_ID << " " << id1 << " " << x1 << " " << id2 << " " << x2 << " " << Q << endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    datafile.close();

    return 0;
}