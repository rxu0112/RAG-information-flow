The base code can be found following [this link](https://github.com/edwinrobots/BayesianOpt_uncertaiNLP2024).

### Training the Passage Utility Predictor

The following it the script to train the Passage Utility predictor. Some flags as we use them are set in the command, the other flags are explained below.

```
python3 passage_utility/main.py
   --epochs 3
   --batch_size 32
   --do_train True
   --save_dir HOMEOUT
   --lr_init 2e-5
   --stop_epochs 2
   --wd 0.001
   --model_name RANKER_NAME
   --input_file  HOMEOUT/PASSAGE_LLM_JUDEGEMENTS
   --reference_rank REFRANK
   --combine_loss \"be\"
   --weight_aux w_aux
   --weight_rank w_rank
   --model_select MODEL_SELECTION
   --proportion 0.5
   --num_shards 0
   --add_title True
```

- ```PASSAGE_LLM_JUDEGEMENTS``` is the file annotated with utility judgements by the target LLM model (these are generated with the scripts in the [retrieval_qa/](../retrieval_qa/) folder).
- ```REFRANK``` is the utility criteria variant (see argument specification for accepted values) the one used in the paper is <b>acc_LM-nli</b>.
- ```w_aux``` is the weight given to the BCE loss term and ```w_rank``` to the pariwise ranking term.


For prediction we use the following (we repeat the flags to crate a descriptive out file name -- TODO: should better save and load this):

```
python3 passage_utility/main.py
   --test_mode \"test\"
   --do_test True
   --save_dir HOMEOUT
   --model_name $ranker
   --input_file $HOMEOUT/PASSAGE_LLM_JUDEGEMENTS
   --output_pred_utilities True
   --model_select MODEL_SELECTION
   --reference_rank REFRANK
   --combine_loss \"be\"
   --weight_aux w_aux
   --weight_rank w_rank
   --add_title True
```

