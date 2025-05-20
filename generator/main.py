# V1

from generator.files import train


def main():
    train.main_click_free(
        outdir="generator/files/training-runs/",
        cfg="fastgan",
        data="dataset/artbench256_10k.zip",
        gpus=1,
        batch=10,
        cond=False,
        mirror=False,
        resume="generator/files/church.pkl",
        batch_gpu=None,
        cbase=32768,
        cmax=512,
        glr=None,
        dlr=0.002,
        map_depth=None,
        desc=None,
        metrics=["no_metric"],
        kimg=350,
        tick=4,
        snap=5,
        seed=314453,
        fp32=False,
        nobench=False,
        workers=3,
        dry_run=False,
        restart_every=9999999
    )


if __name__ == "__main__":
    main()
