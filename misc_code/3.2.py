

path1 = "/scratch/peilab/ysunem/26.2/2.27_CausVid/data/mixkit_latents_lmdb/data.mdb"


import lmdb

# 尝试打开目录（LMDB 通常指向包含 data.mdb 的文件夹）
env = lmdb.open('/scratch/peilab/ysunem/26.2/2.27_CausVid/data/mixkit_latents_lmdb', readonly=True)
with env.begin() as txn:
    # 打印前 5 个 Key 看看是什么
    cursor = txn.cursor()
    for i, (key, value) in enumerate(cursor):
        print(f"Key: {key}")
        if i > 5: break
env.close()