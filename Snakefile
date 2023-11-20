rule make_datasets:
    input:
    params:
        bkg = '/eos/home-e/egovorko/kd_data/datasets_-1.npz',
        bkg_ids = '/eos/home-e/egovorko/kd_data/background_IDs_-1.npz',
        anomaly = '/eos/home-e/egovorko/kd_data/bsm_datasets_-1.npz',
        divisions = [0.30, 0.30, 0.20, 0.20]
    output:
        scaling = 'output/scaling.npz',
        dataset = 'output/dataset.npz'
    shell:
        'python src/data/make_datasets.py {params.bkg} {params.bkg_ids} {params.anomaly} \
            --scaling-file {output.scaling} \
            --divisions {params.divisions} \
            --output-filename {output.dataset} \
            --sample-size -1 \
            --anomaly-size -1 '

rule train_cl:
    input:
        data = rules.make_datasets.output.dataset
    output:
    shell:
        'python src/cl/train.py'

rule train_nf:
    input:
    output:
    shell:
        'python src/nf/train.py'

rule plot:
    input:
    output:
