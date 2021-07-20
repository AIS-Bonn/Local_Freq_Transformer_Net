# Local_Freq_Transformer_Net

The code for the paper: "Local Frequency Domain Transformer Networks for Video Prediction"

If you use the code for your research paper, please cite the following paper:
<p>
  Hafez Farazi<b></b>, Jan Nogga , and Sven Behnke:<br>
  <a href="https://arxiv.org/pdf/2105.04637.pdf"><u>Local Frequency Domain Transformer Networks for Video Prediction</u></a>&nbsp;<a href="https://arxiv.org/pdf/2105.04637.pdf">[PDF]</a><br>
  Accepted for International Joint Conference on Neural Networks (IJCNN), July 2021. <br><b></b><br>
</p>

## Dependencies
The code was tested with Ubuntu 18.04 and PyTorch 1.6

## Run
Get a sample trained weights for "Color Moving MNIST on STL" from [here](https://drive.google.com/file/d/1-foTNMURdmDCC-nl3mrVDJ0HDvWfqY05/view?usp=sharing).\
Put it in savedModels folder

```
python app.py --concat_depth=4 --data_key=MotionSegmentation --digitCount=2 --init_A_with_T=True --init_T_with_GT=False --lft_mode=no_crop --max_result_speed=4 --pos_encoding=True --refine_filter_size=5 --refine_hidden_unit=4 --refine_layer_cnt=5 --res_x=129 --res_y=129 --seeAtBegining=3 --sequence_length=10 --sequence_seed=5 --start_T_index=2 --stride=8 --tr_non_lin=PReLU --tran_filters=3333333 --tran_hidden_unit=8 --trans_mode=Conv --untilIndex=8 --use_energy=True --use_variance=True --window_size=18 --window_type=Gaussian --PD_model_enable=True --dublicate_PD_from_tr=False --PD_model_filters=133 --PD_model_non_lin=PReLU --PD_model_hidden_unit=8 --refine_layer_cnt_a=6 --tqdm=True --PD_model_use_direction=False --color=True --load_model=STLMC --inference=True --batch_size=10
```
