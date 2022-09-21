# PFENet

This is implementaion of the paper [**PFENet: Prior Guided Feature Enrichment Network for Few-shot Segmentation**](http://arxiv.org/abs/2008.01449) on wheat rust dataset.

### Environment

- torch==1.4.0 (torch version >= 1.0.1.post2 should be okay to run this repo)
- numpy==1.18.4
- cv2==4.2.0

### Set up

Download pretrained model from [**here**](https://mycuhk-my.sharepoint.com/personal/1155122171_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155122171%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FPFENet%20TPAMI%20Submission%2FPFENet%5Fcheckpoints%2Fpascal%5Fckpt%5F50%2Ezip&parent=%2Fpersonal%2F1155122171%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FPFENet%20TPAMI%20Submission%2FPFENet%5Fcheckpoints&ga=1) <br>
Download pretrained backbone from [**here**](https://mycuhk-my.sharepoint.com/personal/1155122171_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155122171%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FPFENet%20TPAMI%20Submission%2FPFENet%5Fcheckpoints%2Fbackbone%2Ezip&parent=%2Fpersonal%2F1155122171%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FPFENet%20TPAMI%20Submission%2FPFENet%5Fcheckpoints&ga=1) <br>
Place some patches in test data path and some patches in support data path. <br>
Place backbone network in **initmodel** directory.<br>
Give the weights path of PFENet model in config file "pfenet.yaml".<br>
Run **test_pfenet.py**.<br>
The dataloader will take all the patches one by one in test path, and take random support samples(based on number of shots) from support path

### Dataset
[**Link**](https://drive.google.com/file/d/1RCWvtiNe1uqbDqEry8HsVoLh24eQqNyJ/view?usp=sharing) for patches
