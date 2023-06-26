n_acqs=100
for ep in rosenb; do  # qbranin, styb
  for dim in 100; do
    for seed in 117 118 119 120 121 122 123 124 125 126; do
      # NOTE: Other parametrizations and priors are possible. See BoTorchSettings.
      # BASELINES
      # matern ard stationary with constant mean (S)
      python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMconst,Ksk -acq_mode "Fei,NM$n_acqs"
      # stationary with quadratic mean with fixed center/anchor (S+QM)
      python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMquadfhs,Ksk -acq_mode "Fei,NM$n_acqs"
      # stationary + greedy trust region (S+TR)
      python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMconst,Ksk -acq_mode "Ftrgreedyei,NM$n_acqs"
      # nonstationary cylindrical kernel with constant mean (C)
      python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMconst,Kck -acq_mode "Fei,NM$n_acqs"
      # PROPOSED
      # informative kernel with fixed anchor and constant mean (I+X0)
      python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMconst,Kik,KMfixed -acq_mode "Fei,NM$n_acqs" -pool true -nsl s -nsr kii0.1
      # informative kernel with greedy anchor and constant mean (I+XA)
      python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMconst,Kik,KMgreedy -acq_mode "Fei,NM$n_acqs" -pool true -nsl s -nsr kii0.1
      # informative kernel with greedy anchor, constant mean and greedy trust region (I+XA+TR)
      python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMconst,Kik,KMgreedy -acq_mode "Ftrgreedyei,NM$n_acqs" -pool true -nsl s -nsr kii0.1
      # ADDITIONAL METHODS
      # informative kernel with greedy anchor and constant mean but fixed nonstationary lengthscales + ratio (I+XA+F)
      # python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMconst,Kik,KMgreedy -acq_mode "Fei,NM$n_acqs" -pool true -nsl 0.1 -nsr 0.1
      # I+XA+F + greedy trust region (I+XA+F+TR)
      # python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMconst,Kik,KMgreedy -acq_mode "Ftrgreedyei,NM$n_acqs" -pool true -nsl 0.1 -nsr 0.1
      # informative kernel with greedy anchor and quadratic mean with fixed center (I+XA+QM)
      # python bo.py -seed "$seed" -train_mode TMlog -prior_mode Xcenter -eval_mode E"$ep",D"$dim" -model_mode MMquadfhs,Kik,KMgreedy -acq_mode "Fei,NM$n_acqs" -pool true -nsl s -nsr kii0.1
    done
  done
done