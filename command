# Training
python -u run.py \
  --is_training 1 \
  --data custom \
  --root_path ./dataset/ \
  --data_path /home/tupham/Downloads/updated_traffic_data_formatted.csv \
  --model FEDformer \
  --batch_size 256 \
  --train_epochs 10 \
  --patience 3 \
  --freq='t' \
  --detail_freq='t' \
  --seq_len 96 \
  --label_len 48 \
  --pred_len=96 \
  --features S \
  --target Vehicles \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --itr 2 \
  --do_predict


 # Testing with ouput vehicle data count from YOLO
 python -u run.py \
    --is_training 0 \
    --task_id test \
    --root_path ./dataset/ \
    --data_path updated_traffic_data_formatted.csv \
    --checkpoints ./checkpoints/test_FEDformer_ETTh1_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0 \
    --model FEDformer \
    --freq='t' \
    --detail_freq='t' \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --features S \
    --target Vehicles \
    --e_layers 2 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --do_predict

    checkpoints/test_FEDformer_random_modes64_custom_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/

      python -u run.py \
    --is_training 0 \
    --task_id test \
    --root_path ./dataset/ \
    --data_path updated_traffic_data_formatted.csv \
    --data='ETTm1' \
    --model FEDformer \
    --batch_size 128 \
    --train_epochs 3 \
    --patience 3 \
    --freq='t' \
    --detail_freq='t' \
    --seq_len 6 \
    --label_len 3 \
    --pred_len 24 \
    --features S \
    --target Vehicles \
    --e_layers 2 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 1 \
    --dec_in 1 \
    --d_model 512 \
    --itr 1 \
    --do_predict




python forecast.py \
  --do_predict \
  --use_gpu \
  --checkpoint /home/tupham/Documents/Development/FEDformer/FEDformer/checkpoints/23jun_01_21am/checkpoint.pth \
  --is_training 0 \
  --modes 64 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1
