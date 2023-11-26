���      }�(�args��argparse��	Namespace���)��}�(�lr�G?PbM����	max_state�K�bz�M��epoch�K�seq_max_len�K�
early_stop�K�run��	integrate��seed�K �
activation��none��	train_fsa�K�train_wildcard�K �train_linear�K �train_V_embed�K �train_word_embed�K �
train_beta�K �beta�G?�      �farnn�K �	bias_init�K�wfa_type��viterbi��clamp_score�K �	optimizer��ADAM��additional_nonlinear�h�additional_state�K �dataset��TREC��train_portion�G?�      �random�K �bidirection�K �normalize_automata��l2��random_noise�G?PbM����	embed_dim�Kd�automata_path_forward��F../data/TREC/automata/automata.split.randomseed.200.False.0.0021.0.pkl��automata_path_backward��P../data/TREC/automata/automata.newrule.reversed.randomseed200.False.0.0354.0.pkl��
model_type��FSARNN�ub�res�]�(G?���l�DG?�C��%G?��x���G?�9XbNe�loggers��src.utils.utils��Logger���)��}��record�]�(�uTRAIN Epoch: 0 | ACC: 0.5613293051359517, LOSS: 0.0031064771333369845, P: 0.5613293051359517, R: 0.5613293051359517 
��LDEV Epoch: 0 | ACC: 0.552, LOSS: 0.0028358356952667236, P: 0.552, R: 0.552 
��MTEST Epoch: 0 | ACC: 0.642, LOSS: 0.0024485313892364502, P: 0.642, R: 0.642 
��sTRAIN INITIAL: ACC: 0.5613293051359517, LOSS: 0.0031064771333369845, P: 0.5613293051359517, R: 0.5613293051359517 
��JDEV INITIAL: ACC: 0.552, LOSS: 0.0028358356952667236, P: 0.552, R: 0.552 
��KTEST INITIAL: ACC: 0.642, LOSS: 0.0024485313892364502, P: 0.642, R: 0.642 
��FTRAIN Epoch: 1 | ACC: 0.5887210473313192, LOSS: 0.002563517067247527 
��LDEV Epoch: 1 | ACC: 0.586, LOSS: 0.0022460157871246338, P: 0.586, R: 0.586 
��FTRAIN Epoch: 2 | ACC: 0.5975830815709969, LOSS: 0.002112754402083813 
��LDEV Epoch: 2 | ACC: 0.626, LOSS: 0.0019955281019210817, P: 0.626, R: 0.626 
��GTRAIN Epoch: 3 | ACC: 0.6219536757301107, LOSS: 0.0021742806333789654 
��LDEV Epoch: 3 | ACC: 0.636, LOSS: 0.0019273751974105836, P: 0.636, R: 0.636 
��ETRAIN Epoch: 4 | ACC: 0.633031218529708, LOSS: 0.001927444461849401 
��LDEV Epoch: 4 | ACC: 0.656, LOSS: 0.0018610801696777343, P: 0.656, R: 0.656 
��GTRAIN Epoch: 5 | ACC: 0.6725075528700907, LOSS: 0.0018450316465633274 
��IDEV Epoch: 5 | ACC: 0.65, LOSS: 0.0018498610258102417, P: 0.65, R: 0.65 
��FTRAIN Epoch: 6 | ACC: 0.6704934541792548, LOSS: 0.001939496677087513 
��IDEV Epoch: 6 | ACC: 0.67, LOSS: 0.0019315991401672363, P: 0.67, R: 0.67 
��GTRAIN Epoch: 7 | ACC: 0.6862034239677745, LOSS: 0.0018477503506678708 
��LDEV Epoch: 7 | ACC: 0.674, LOSS: 0.0018127957582473756, P: 0.674, R: 0.674 
��GTRAIN Epoch: 8 | ACC: 0.7015105740181269, LOSS: 0.0017826932314539484 
��KDEV Epoch: 8 | ACC: 0.672, LOSS: 0.001782149076461792, P: 0.672, R: 0.672 
��GTRAIN Epoch: 9 | ACC: 0.6972809667673716, LOSS: 0.0017032385592734346 
��JDEV Epoch: 9 | ACC: 0.694, LOSS: 0.00173629629611969, P: 0.694, R: 0.694 
��HTRAIN Epoch: 10 | ACC: 0.7061430010070493, LOSS: 0.0016461526879371113 
��MDEV Epoch: 10 | ACC: 0.692, LOSS: 0.0017187787294387817, P: 0.692, R: 0.692 
��HTRAIN Epoch: 11 | ACC: 0.7043303121852971, LOSS: 0.0020650982136452664 
��MDEV Epoch: 11 | ACC: 0.692, LOSS: 0.0016756821870803833, P: 0.692, R: 0.692 
��GTRAIN Epoch: 12 | ACC: 0.7057401812688822, LOSS: 0.002653388554476059 
��LDEV Epoch: 12 | ACC: 0.688, LOSS: 0.001849881649017334, P: 0.688, R: 0.688 
��HTRAIN Epoch: 13 | ACC: 0.6876132930513595, LOSS: 0.0030695723142748752 
��MDEV Epoch: 13 | ACC: 0.686, LOSS: 0.0017488702535629272, P: 0.686, R: 0.686 
��HTRAIN Epoch: 14 | ACC: 0.6892245720040282, LOSS: 0.0017037735844906963 
��MDEV Epoch: 14 | ACC: 0.674, LOSS: 0.0017576004266738891, P: 0.674, R: 0.674 
��HTRAIN Epoch: 15 | ACC: 0.7023162134944613, LOSS: 0.0034811237667502475 
��LDEV Epoch: 15 | ACC: 0.686, LOSS: 0.001952563524246216, P: 0.686, R: 0.686 
��HTRAIN Epoch: 16 | ACC: 0.7037260825780464, LOSS: 0.0016777329454489224 
��LDEV Epoch: 16 | ACC: 0.692, LOSS: 0.001977243661880493, P: 0.692, R: 0.692 
��HTRAIN Epoch: 17 | ACC: 0.6998992950654582, LOSS: 0.0016224608920732174 
��MDEV Epoch: 17 | ACC: 0.688, LOSS: 0.0018864606618881226, P: 0.688, R: 0.688 
��HTRAIN Epoch: 18 | ACC: 0.7013091641490433, LOSS: 0.0015733935321565842 
��MDEV Epoch: 18 | ACC: 0.686, LOSS: 0.0018556938171386719, P: 0.686, R: 0.686 
��HTRAIN Epoch: 19 | ACC: 0.7075528700906344, LOSS: 0.0015434984954342261 
��MDEV Epoch: 19 | ACC: 0.684, LOSS: 0.0018233081102371216, P: 0.684, R: 0.684 
��HTRAIN Epoch: 20 | ACC: 0.7105740181268883, LOSS: 0.0015104677261782798 
��LDEV Epoch: 20 | ACC: 0.686, LOSS: 0.001798043727874756, P: 0.686, R: 0.686 
��GTRAIN Epoch: 21 | ACC: 0.7115810674723061, LOSS: 0.001489464814089096 
��MDEV Epoch: 21 | ACC: 0.688, LOSS: 0.0017657490968704223, P: 0.688, R: 0.688 
��HTRAIN Epoch: 22 | ACC: 0.7137965760322256, LOSS: 0.0014732754842752417 
��GDEV Epoch: 22 | ACC: 0.7, LOSS: 0.0017448679208755494, P: 0.7, R: 0.7 
��HTRAIN Epoch: 23 | ACC: 0.7180261832829808, LOSS: 0.0014566106738640823 
��MDEV Epoch: 23 | ACC: 0.706, LOSS: 0.0017286466360092163, P: 0.706, R: 0.706 
��HTRAIN Epoch: 24 | ACC: 0.7280966767371602, LOSS: 0.0014407384431614255 
��KDEV Epoch: 24 | ACC: 0.708, LOSS: 0.00171491539478302, P: 0.708, R: 0.708 
��HTRAIN Epoch: 25 | ACC: 0.7303121852970795, LOSS: 0.0014280125933951603 
��MDEV Epoch: 25 | ACC: 0.706, LOSS: 0.0017057477235794068, P: 0.706, R: 0.706 
��HTRAIN Epoch: 26 | ACC: 0.7353474320241692, LOSS: 0.0014163782709434793 
��MDEV Epoch: 26 | ACC: 0.708, LOSS: 0.0016997798681259156, P: 0.708, R: 0.708 
��HTRAIN Epoch: 27 | ACC: 0.7377643504531722, LOSS: 0.0014095760304161909 
��MDEV Epoch: 27 | ACC: 0.712, LOSS: 0.0016869391202926636, P: 0.712, R: 0.712 
��GTRAIN Epoch: 28 | ACC: 0.7393756294058409, LOSS: 0.001396066340554997 
��MDEV Epoch: 28 | ACC: 0.716, LOSS: 0.0016815541982650756, P: 0.716, R: 0.716 
��HTRAIN Epoch: 29 | ACC: 0.7375629405840887, LOSS: 0.0013911447376162864 
��MDEV Epoch: 29 | ACC: 0.712, LOSS: 0.0016697419881820678, P: 0.712, R: 0.712 
��HTRAIN Epoch: 30 | ACC: 0.7413897280966767, LOSS: 0.0013790730503270513 
��MDEV Epoch: 30 | ACC: 0.712, LOSS: 0.0016682978868484497, P: 0.712, R: 0.712 
��NTEST Epoch: 30 | ACC: 0.772, LOSS: 0.0013640848398208617, P: 0.772, R: 0.772 
�esbu.