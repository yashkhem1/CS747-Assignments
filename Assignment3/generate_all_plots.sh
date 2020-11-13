#!/bin/bash
echo "4 moves without stochasticity -- SARSA"
python3 generate_plots.py --timesteps 25000 --algorithms "SARSA" --title "4 moves without stochasticity" --colors g --outfile sarsa_4.png --min_policy --heatmap
echo "8 moves without stochasticity -- SARSA"
python3 generate_plots.py --timesteps 15000 --algorithms "SARSA" --title "8 moves without stochasticity" --colors g --outfile sarsa_8.png --min_policy --heatmap --eight_moves
echo "8 moves with stochasticity -- SARSA"
python3 generate_plots.py --timesteps 50000 --algorithms "SARSA" --title "8 moves with stochasticity" --colors g --outfile sarsa_8_stoch.png --heatmap --eight_moves --stochastic
echo "4 moves without stochasticity -- SARSA Expected-SARSA Q-Learning"
python3 generate_plots.py --timesteps 25000 --algorithms "SARSA" "Expected SARSA" "Q Learning" --title "4 moves without stochasticity" --colors g r b --outfile all_4.png
echo "8 moves without stochasticity -- SARSA Expected-SARSA Q-Learning"
python3 generate_plots.py --timesteps 15000 --algorithms "SARSA" "Expected SARSA" "Q Learning" --title "8 moves without stochasticity" --colors g r b --outfile all_8.png --eight_moves
echo "8 moves with stochasticity -- SARSA Expected-SARSA Q-Learning"
python3 generate_plots.py --timesteps 50000 --algorithms "SARSA" "Expected SARSA" "Q Learning" --title "8 moves with stochasticity" --colors g r b --outfile all_8_stoch.png --eight_moves --stochastic