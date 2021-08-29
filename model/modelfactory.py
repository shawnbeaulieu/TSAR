class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, channels=112, nm_channels=192, nm_rep_size=1728, rep_size=112, in_channels=6, treatment='TSAR'):

        if dataset == "omniglot" or dataset == "imagenet" or dataset == 'cifar':
            if treatment == "TSAR":
                return [
                
                    # =============== Regulatory Network ===============


                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]),
                    ('bn1_nm', [nm_channels]),
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn2_nm', [nm_channels]),
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn3_nm', [nm_channels]),

                    ('linear_nm_c1', [channels*3*3*3, nm_rep_size]),
                    ('linear_nm_c2', [channels*channels*3*3, nm_rep_size]),
                    ('linear_nm_c3', [channels*channels*3*3, nm_rep_size]),
                    ('linear_nm_fc', [rep_size*1000, nm_rep_size]),

                    # =============== Classification Network ===============

                    ('conv1', [channels, 3, 3, 3, 1, 0]),
                    ('bn1', [channels]),
                    ('conv2', [channels, channels, 3, 3, 1, 0]),
                    ('bn2', [channels]),
                    ('conv3', [channels, channels, 3, 3, 1, 0]),
                    ('bn3', [channels]),
                    ('linear_out', [1000, rep_size]),
 
                ]

            elif treatment == "ANML":

                return [

                    # =============== Regulatory Network ===============


                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]),
                    ('bn1_nm', [nm_channels]),
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn2_nm', [nm_channels]),
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn3_nm', [nm_channels]),
                    ('linear_nm_fc', [rep_size, nm_rep_size]),

                    # =============== Classification Network ===============

                    ('conv1', [channels, 3, 3, 3, 1, 0]),
                    ('bn1', [channels]),
                    ('conv2', [channels, channels, 3, 3, 1, 0]),
                    ('bn2', [channels]),
                    ('conv3', [channels, channels, 3, 3, 1, 0]),
                    ('bn3', [channels]),
                    ('linear_out', [1000, rep_size]),

                ]



        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
