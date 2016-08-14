## Mars Express Power Challenge ##
<a href="https://kelvins.esa.int/mars-express-power-challenge/">
<img align=right width="297" height="240" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Mars-express-volcanoes-sm.jpg/594px-Mars-express-volcanoes-sm.jpg"></a>

https://kelvins.esa.int/mars-express-power-challenge/

> "The Mars Express Power Challenge focuses on the difficult problem of predicting the thermal power consumption. Three full Martian years of [Mars Express][Mars_Express] telemetry are made available and you are challenged to predict the thermal subsystem power consumption on the following Martian year."


### About ###

This repository contains all the work I did for the 'Mars Express Power' competition. It consists of two files:

 * [echo_state_networks.py](echo_state_networks.py)
 * [workbench.ipynb][nb]

My final submission reached a RMSE of 0.089078627464354 on the [public leaderboard][public leaderboard], and later a RMSE of 0.088395630359812905 on the [final leaderboard][final leaderboard], allowing me to rank in 5th place in the competition.
The file from that final submission is included here ([lfs_submission_5b__rebuilt.csv](lfs_submission_5b__rebuilt.csv)), and can be regenerated as well by running the code in the Jupyter notebook.


### Features ###

See my post in the official forum ([here][end_post]) for an outline of the features I used.


### Models ###

I saw the competition as an opportunity to finally spend some time studying and implementing [Echo State Networks][ESN_schol] (a type of Recurrent Neural Network). All my modelling effort was therefore spent on getting the most I could out of them, and them alone.

The Jupyter notebook's sections on "[Parameter sweeps][nb_params]" and "[Training ensembles][nb_ensem]" describe the steps taken to understand their behaviour, and to train them as accurately as possible.

The implemented [echo_state_networks.py](echo_state_networks.py) tries to follow scikit-learn's interface and naming conventions. See the "[Parameter sweeps][nb_params]" section on the notebook for a description of its parameters.

![Test set error over a 30-days sliding window](plots/error_over_time__testset__per_month.png)
![Test set error in all predictions up to the current date](plots/error_over_time__testset__so_far.png)

### References ###

1. Jaeger, H. (2007). [Echo state network][ESN_schol]. *Scholarpedia*, 2(9), 2330.
2. Lukoševičius, M. (2012). [A practical guide to applying echo state networks][ESN_guide]. In *Neural networks: Tricks of the trade* (pp. 659-686). Springer Berlin Heidelberg.
3. Principe, J. C., & Chen, B. (2015). [Universal approximation with convex optimization: Gimmick or reality?][CULM] *IEEE Computational Intelligence Magazine,* 10(2), 68-77.


### Dependencies ###

The code shared here was written in Python 3 (3.4.4). It has the following dependencies:

 * [echo_state_networks.py](echo_state_networks.py):
   * numpy, scipy, tqdm
 * [workbench.ipynb][nb]:
   * numpy, scipy, matplotlib, seaborn, pandas, scikit-learn, tqdm


### Leaderboard ###

Most of the competition's top ranked players/teams have shared their code. You can find below links to their repositories.

| Rank | Name            | Score                | Repository |
|:----:|:--------------- |:-------------------- |:---------- |
| 1    | MMMe8           | 0.079163638689759466 |            |
| 2    | redrock         | 0.080301894079712499 | [stephanos-stephani/MarsExpressChallenge][repo_2] |
| 3    | fornaxintospace | 0.081925542258189737 |            |
| 4    | Alex            | 0.083848704280679837 | [alex-bauer/kelvin-power-challenge][repo_4] |
| 5    | luis            | 0.088395630359812905 | [lfsimoes/mars_express__esn][repo_5] |
| 6    | w               | 0.088993096282001347 | [wsteitz/mars_express][repo_6] |
| 7    | trnka           | 0.089866726592717425 | [ktrnka/mars-express][repo_7] |

[repo_1]: https://?
[repo_2]: https://github.com/stephanos-stephani/MarsExpressChallenge
[repo_3]: https://?
[repo_4]: https://github.com/alex-bauer/kelvin-power-challenge
[repo_5]: https://github.com/lfsimoes/mars_express__esn
[repo_6]: https://github.com/wsteitz/mars_express
[repo_7]: https://github.com/ktrnka/mars-express




[Mars_Express]: https://en.wikipedia.org/wiki/Mars_Express
[public leaderboard]: https://kelvins.esa.int/mars-express-power-challenge/leaderboard/
[final leaderboard]: https://kelvins.esa.int/mars-express-power-challenge/results/

[end_post]: https://kelvins.esa.int/mars-express-power-challenge/discussion/110/#c115

[ESN_schol]: http://www.scholarpedia.org/article/Echo_state_network
[ESN_guide]: http://minds.jacobs-university.de/sites/default/files/uploads/papers/PracticalESN.pdf
[CULM]: http://dx.doi.org/10.1109/MCI.2015.2405352

[nb]: http://nbviewer.jupyter.org/github/lfsimoes/mars_express__esn/blob/master/workbench.ipynb
[nb_params]: http://nbviewer.jupyter.org/github/lfsimoes/mars_express__esn/blob/master/workbench.ipynb#Parameter-sweep
[nb_ensem]: http://nbviewer.jupyter.org/github/lfsimoes/mars_express__esn/blob/master/workbench.ipynb#Training-ensembles
