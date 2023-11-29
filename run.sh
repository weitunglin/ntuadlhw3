taiwan_llama_path=$1
adapter_path=$2
input_file=$3
output_file=$4

shift 4
flags=$@
echo $flags

python3 test.py --base_model_path ${taiwan_llama_path} --peft_path ${adapter_path} --test_data_path ${input_file} --output_path ${output_file} $flags
