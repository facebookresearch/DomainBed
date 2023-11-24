import argparse
import numpy as np
from PIL import Image
import torch
from domainbed import algorithms
import scipy.io as scio
import os
import torch.nn.functional as F

#读取网络
#下为原程序生成的model.pkl所存储的东西:
# save_dict = {
#     "args": vars(args),
#     "model_input_shape": dataset.input_shape,
#     "model_num_classes": dataset.num_classes,
#     "model_num_domains": len(dataset) - len(args.test_envs),
#     "model_hparams": hparams,
#     "model_dict": algorithm.state_dict()
# }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw Hyperspectral Classification Map')
    parser.add_argument('--algorithm_dict', type=str,default='./train_output/model.pkl',
                        help='the path of the saving parameters')
    parser.add_argument('--path', type=str,default='./domainbed/data/Indian_pines',
                        help='the path of 3d-')
    parser.add_argument('--out_path', type=str,default='./train_output',
                        help='the path of 3d-')
    parser.add_argument('--out_name', type=str,default='erm',
                        help='the path of 3d-')
    args = parser.parse_args()
    pre_hyperparameters = torch.load(args.algorithm_dict)
    print(pre_hyperparameters['args'])

    algorithm_class = algorithms.get_algorithm_class(pre_hyperparameters['args']['algorithm'])#返回一个algorithm对象(尚未初始化)
    algorithm = algorithm_class(pre_hyperparameters['model_input_shape'], pre_hyperparameters['model_num_classes'],#初始化
        pre_hyperparameters['model_num_domains'], pre_hyperparameters['model_hparams'])#倒数第二个是num_of_domains 

    algorithm.load_state_dict(pre_hyperparameters['model_dict'])     
    algorithm.to("cuda")
    print(algorithm.predict(torch.tensor(np.ones(200))).argmax(1).cpu().numpy()[0])

    #label颜色对照字典
    dic = {0 : np.array([0,134,139]),
           1 : np.array([106,90,205]),
           2 : np.array([154,255,154]),
           3 : np.array([218,165,32]),
           4 : np.array([0,0,255])}

    def label_change(num):
        label_change_dictionary = {
            0 : 0,
            1 : 0,
            2 : 0,
            3 : 1,
            4 : 0,
            5 : 2,
            6 : 3,
            7 : 0,
            8 : 0,
            9 : 0,
            10: 4,
            11: 5,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0
        }
        return label_change_dictionary.get(num, None)
    
    data = scio.loadmat(os.path.join(args.path,'Indian_pines_corrected'))['indian_pines_corrected']
    gt = scio.loadmat(os.path.join(args.path,'Indian_pines_gt'))['indian_pines_gt']

    #测试用
    # print(data[0][0].shape)
    # print(torch.tensor(data[0][0].astype(np.float64)).shape)
    #先搞一张原始3维照片
    pic = np.zeros([gt.shape[0],gt.shape[1],3])

    def draw_model(no):
        for i in list(range(gt.shape[0])):
            for j in list(range(gt.shape[1])):
                if gt[i][j] in [0,1,2,4,7,8,9,12,13,14,15,16]:
                    pic[i][j] = np.array([248,248,255])#无关元素赋灰白
                else:
                    #print(model.predict(data[i][j].reshape(-1, 102)))
                    pic[i][j] = dic[algorithm.predict(torch.tensor(data[i][j].astype(np.float64))).argmax(1).cpu().numpy()[0]]
                    
        img = Image.fromarray(np.uint8(pic)).convert('RGB')
        img.save(os.path.join(args.out_path,'IP_out_{}.bmp'.format(no)))

    draw_model(args.out_name)


