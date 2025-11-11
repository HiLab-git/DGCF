from train import train
from test import test
from evaluation import evaluation
from code_config.parser import parse_json_file

def main():
    config_root = "./file_config"
    # 将config_root下的所有配置文件保存到副本中
    configs = {}
    import os
    for file in os.listdir(config_root):
        configs[file] = parse_json_file(os.path.join(config_root,file))

    do_train = True
    do_test = True
    do_eval = True
    
    train_output = None
    test_output = None
    
    if do_train:
        print("Starting training...")
        train_output = train(status_config=configs["train.json"])
        print(f"Training completed. ")
    
    if do_test:
        print("Starting testing...")
        test_output = test(status_config=configs["test.json"], common_config=train_output)
        print(f"Testing completed. ")
    
    if do_eval:
        print("Starting evaluation...")
        eval_output = evaluation(status_config=configs["evaluation.json"], common_config=test_output)
        print(f"Evaluation completed. ")

if __name__ == "__main__":
    main()