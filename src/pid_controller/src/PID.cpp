#include <iostream>
#include "PID.h"
/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}


void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
    time_step += 1;
    std::cout << "\n\n\n# ------------------------------------------------------------------------------" << "\n";
    std::cout<<"# [Timestep] =  " << time_step << "\n";
    std::cout << "# ------------------------------------------------------------------------------" << "\n";

    d_error = cte - p_error;
    p_error = cte;
    i_error += cte;
    total_cte += cte;

    std::cout << "[Before]:" << "\n";
    std::cout << "\tcte = " << cte << "\ttotal_cte = " << total_cte << "\n";
    std::cout << "\tp_error = " << p_error << "\ti_error = " << i_error << "\td_error = " << d_error  << "\n";
    std::cout << "\tdkp = " << dkp << "\tdki = " << dki << "\tdkd = " << dkd << "\n";
    std::cout << "\tKp = " << Kp << "\tKi = " << Ki << "\tKd = " << Kd << "\n";

    std::cout << "[After]: "<< "\n";
    if (time_step > 1 && cte>0.00001){
        int param_num =  time_step % 3;
        if (param_num == 0){
            vector<double> param_deltas = twiddleIt(Kp, dkp);
            Kp = param_deltas[0];
            dkp = param_deltas[1];
        }
        else if(param_num == 1){
            vector<double> param_deltas = twiddleIt(Ki, dki);
            Ki = param_deltas[0];
            dki = param_deltas[1];
        }
        else if(param_num == 2){
            vector<double> param_deltas = twiddleIt(Kd, dkd);
            Kd = param_deltas[0];
            dkd = param_deltas[1];
        }
        else{
            exit (EXIT_FAILURE);
        }
        std::cout << "\tUpdate for = " << param_num << "\n";
    }

    std::cout << "\tdkp = " << dkp << "\tdki = " << dki << "\tdkd = " << dkd << "\n";
    std::cout << "\tKp = " << Kp << "\tKi = " << Ki << "\tKd = " << Kd << "\n";
}

vector<double> PID::twiddleIt(double param, double delta){

    double curr_error = total_cte/time_step;
    std::cout << "[TWIDDLE IT] \n";
    if (
        tried_decrementing == 1 &&
        tried_incrementing == 1 &&
        curr_error>=least_error
    ){
        /*
        * Activated: When increasing param and decreasing param by delta does not work
        * we set param to the original_value and decrease our deltas
        */
        param += delta; // We add to param because in the previous step we had tested the error by decreasing
        delta *= 0.9;
        tried_incrementing = 0;
        tried_decrementing = 0;
        std::cout << "[Decrementing delta] -> " << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
    }
    else{
        if (curr_error<least_error){
            // If the error goes down we store it and
            least_error = curr_error;
            delta *= 1.1;
            tried_incrementing = 0;
            tried_decrementing = 0;
        }
        else{
            if (tried_incrementing == 0){
                // When I havent tried incrementing, I increment
                param += delta;
                tried_incrementing = 1;
                std::cout << "[Incrementing delta] -> inc_param" << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
            }
            else if (tried_decrementing == 0){
                // When I have tried incrementing but not decrementing I decrement
                // We multiply by 2 because in the previous step we incremented by 2
                param -= 2*delta;
                tried_decrementing = 1;
                std::cout << "[Increment delta] -> dec_param" << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
            }
        }
    }

    vector<double> params_deltas = {param, delta};
    return params_deltas;
}

double PID::calculateSteeringValue(){
    double steering_val = -Kp*p_error - Kd*d_error - Ki*i_error;
    return steering_val;
}