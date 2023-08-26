# field_electron_emission
This directory contains the code and selected DFT results for the paper "Modeling field electron emission from a flat Au (100) surface with density-functional theory."
"WKB_emission.py" generates the emission current densities using the classical Fowler-Nordheim framework.
"exact_QM_transferMatrix.py" generates the emission current densities using the transfer-matrix method for transmission coefficients. 
"exact_QM_RungeKutta.py" generates the transmission coefficient using a selected Runge-Kutta method.
"repulsion_distance_calculator.py" finds the electrical surface using the induced charge densities.
"elec_surf_comp.py" compares the electrical surfaces found with different techniques.
"repulsion_distance_average.py" finds the average repulsion distance based on the selected comparison method.
