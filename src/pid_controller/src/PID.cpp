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

    if (time_step == 1) {
        // to get correct initial d_error
        p_error = cte;
    }
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
//    if (time_step > 1 && cte>0.00001){
//        int param_index =  time_step % 3;
    if (param_index == 0){
        vector<double> param_deltas = twiddleIt(Kp, dkp);
        Kp = param_deltas[0];
        dkp = param_deltas[1];
    }
    else if(param_index == 1){
        vector<double> param_deltas = twiddleIt(Ki, dki);
        Ki = param_deltas[0];
        dki = param_deltas[1];
    }
    else if(param_index == 2){
        vector<double> param_deltas = twiddleIt(Kd, dkd);
        Kd = param_deltas[0];
        dkd = param_deltas[1];
    }
    else{
        exit (EXIT_FAILURE);
    }
    std::cout << "\tUpdate for = " << param_index << "\n";
//    }

    std::cout << "\tdkp = " << dkp << "\tdki = " << dki << "\tdkd = " << dkd << "\n";
    std::cout << "\tKp = " << Kp << "\tKi = " << Ki << "\tKd = " << Kd << "\n";
}

vector<double> PID::twiddleIt(double param, double delta){

    double curr_error = total_cte/time_step;
    std::cout << "[TWIDDLE IT] param_index = " << param_index <<  " curr_error = " << curr_error << " least_error = " << least_error << "\n";
    if (
        tried_decrementing == 1 &&
        tried_incrementing == 1 &&
        curr_error>least_error
    ){
        /*
        * Activated: When increasing param and decreasing param by delta does not work
        * we set param to the original_value and decrease our deltas
        */
        param += delta; // We add to param because in the previous step we had tested the error by decreasing
        delta *= 0.9;
        param_index = (param_index + 1) % 3; // Change param when an resolve was made on either increasing or decreasing delta
        tried_incrementing = 0;
        tried_decrementing = 0;
        std::cout << "[Decrementing delta] -> " << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
    }
    else{
        if (curr_error<least_error){
            // If the error goes down we store it and
            least_error = curr_error;
            delta *= 1.1;
            param_index = (param_index + 1) % 3; // Change param when an resolve was made on either increasing or decreasing delta
            tried_incrementing = 0;
            tried_decrementing = 0;
        }
        else{
            if (tried_incrementing == 0){
                // When I havent tried incrementing, I increment
                param += delta;
                tried_incrementing = 1;
                std::cout << "[Incrementing param]" << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
            }
            else if (tried_decrementing == 0){
                // When I have tried incrementing but not decrementing I decrement
                // We multiply by 2 because in the previous step we incremented by 2
                param -= 2*delta;
                tried_decrementing = 1;
                std::cout << "['Decrementing Param]" << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
            }
        }
    }

    vector<double> params_deltas = {param, delta};
    return params_deltas;
}



//vector<double> PID::twiddleIt(double param, double delta){
//
//    double curr_error = total_cte/time_step;
//    std::cout << "[TWIDDLE IT] param_index = " << param_index <<  " curr_error = " << curr_error << " least_error = " << least_error << "\n";
////    if (
////        tried_decrementing == 1 &&
////        tried_incrementing == 1 &&
////        curr_error>=least_error
////    ){
////        /*
////        * Activated: When increasing param and decreasing param by delta does not work
////        * we set param to the original_value and decrease our deltas
////        */
////        param += delta; // We add to param because in the previous step we had tested the error by decreasing
////        delta *= 0.9;
////        tried_incrementing = 0;
////        tried_decrementing = 0;
////        std::cout << "[Decrementing delta] -> " << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
////    }
//    if (curr_error<least_error){
//        // If the error goes down we store it and
//        least_error = curr_error;
//        delta *= 1.1;
//        param_index = (param_index + 1) % 3;
//        tried_incrementing = 0;
//        tried_decrementing = 0;
//    }
//    if (tried_incrementing == 0 && tried_decrementing == 0){
//        // When I havent tried incrementing, I increment
//        param += delta;
//        tried_incrementing = 1;
//        std::cout << "[Incrementing delta] -> inc_param" << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
//    }
//    else if (tried_incrementing == 1 && tried_decrementing == 0){
//        // When I have tried incrementing but not decrementing I decrement
//        // We multiply by 2 because in the previous step we incremented by 2
//        param -= 2*delta;
//        tried_decrementing = 1;
//        std::cout << "[Decrement delta] -> dec_param" << "\tnew_param = " << param << "\tnew_delta = " << delta << "\n";
//    }
//    else{
//        param += delta;
//        delta *= 0.9;
//        param_index = (param_index + 1) % 3;
//        tried_incrementing = 0;
//        tried_decrementing = 0;
//    }
//
//    vector<double> params_deltas = {param, delta};
//    return params_deltas;
//}

double PID::calculateSteeringValue(){
    double steering_val = -Kp*p_error - Kd*d_error - Ki*i_error;
    return steering_val;
}


vector<double> PID::getParams(){
    vector<double> out_params;
    out_params = {Kp, Ki, Kd, p_error, i_error, d_error, dkp, dki, dkd};
    return out_params;
}






if (twiddle == true){
    total_cte = total_cte + pow(cte,2);
    if(n==0){
      pid.Init(p[0],p[1],p[2]);
    }
    //Steering value
    pid.UpdateError(cte);
    steer_value = pid.TotalError();
    // DEBUG
    //std::cout << "CTE: " << cte << " Steering Value: " << steer_value << " Throttle Value: " << throttle_value << " Count: " << n << std::endl;
    n = n+1;
    if (n > max_n){
      //double sump = p[0]+p[1]+p[2];
      //std::cout << "sump: " << sump << " ";
      if(first == true) {
        std::cout << "Intermediate p[0] p[1] p[2]: " << p[0] << " " << p[1] << " " << p[2] << " ";
        p[p_iterator] += dp[p_iterator];
        //pid.Init(p[0], p[1], p[2]);
        first = false;
      }else{
        error = total_cte/max_n;
        if(error < best_error && second == true) {
            best_error = error;
            best_p[0] = p[0];
            best_p[1] = p[1];
            best_p[2] = p[2];
            dp[p_iterator] *= 1.1;
            sub_move += 1;
            std::cout << "iteration: " << total_iterator << " ";
            std::cout << "p_iterator: " << p_iterator << " ";
            std::cout << "p[0] p[1] p[2]: " << p[0] << " " << p[1] << " " << p[2] << " ";
            std::cout << "error: " << error << " ";
            std::cout << "best_error: " << best_error << " ";
            std::cout << "Best p[0] p[1] p[2]: " << best_p[0] << " " << best_p[1] << " " << best_p[2] << " ";
        }else{
          //std::cout << "else: ";
          if(second == true) {
            std::cout << "Intermediate p[0] p[1] p[2]: " << p[0] << " " << p[1] << " " << p[2] << " ";
            p[p_iterator] -= 2 * dp[p_iterator];
            //pid.Init(p[0], p[1], p[2]);
            second = false;
          }else {
            std::cout << "iteration: " << total_iterator << " ";
            std::cout << "p_iterator: " << p_iterator << " ";
            std::cout << "p[0] p[1] p[2]: " << p[0] << " " << p[1] << " " << p[2] << " ";
            if(error < best_error) {
                best_error = error;
                best_p[0] = p[0];
                best_p[1] = p[1];
                best_p[2] = p[2];
                dp[p_iterator] *= 1.1;
                sub_move += 1;
            }else {
                p[p_iterator] += dp[p_iterator];
                dp[p_iterator] *= 0.9;
                sub_move += 1;
            }
            std::cout << "error: " << error << " ";
            std::cout << "best_error: " << best_error << " ";
            std::cout << "Best p[0] p[1] p[2]: " << best_p[0] << " " << best_p[1] << " " << best_p[2] << " ";
          }
        }
      }
      if(sub_move > 0) {
        p_iterator = p_iterator+1;
        first = true;
        second = true;
        sub_move = 0;
      }
      if(p_iterator == 3) {
        p_iterator = 0;
      }
      total_cte = 0.0;
      n = 0;
      total_iterator = total_iterator+1;
      double sumdp = dp[0]+dp[1]+dp[2];
      if(sumdp < tol) {
        //pid.Init(p[0], p[1], p[2]);
        std::cout << "Best p[0] p[1] p[2]: " << best_p[0] << best_p[1] << best_p[2] << " ";
        //ws.close();
        //std::cout << "Disconnected" << std::endl;
      } else {
        std::string reset_msg = "42[\"reset\",{}]";
        ws.send(reset_msg.data(), reset_msg.length(), uWS::OpCode::TEXT);
      }
    } else {
      msgJson["steering_angle"] = steer_value;
      msgJson["throttle"] = throttle_value;
      auto msg = "42[\"steer\"," + msgJson.dump() + "]";
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
    }
  }