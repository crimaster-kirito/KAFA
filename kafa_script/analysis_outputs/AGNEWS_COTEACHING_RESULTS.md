# AG News Co-Teaching strong-baseline development result

**Status: frozen development-only comparison; no official test labels used.**

## Verification

- Protocol SHA-256: `465aa3409a1b17fe7841c0bfc870d71a6fd0222815345e0ef5b52bfdc42db365`.
- All 54 records evaluate the same frozen 800-example development split at seeds 75--77.
- Every naive/Co-Teaching pair has an identical mixture-assignment hash.
- Six contaminated family-by-condition tests are Holm-adjusted together; Co-Teaching is advantaged by the true contamination fraction as its forget rate.

## Accuracy

| Client | Condition | Naive | Co-Teaching ensemble | Paired gain [95% CI] | Raw p | Holm p |
|---|---|---:|---:|---:|---:|---:|
| qwen | clean | 86.67% | 86.92% | +0.25 pp [-3.95, +4.45] | -- | -- |
| qwen | conflict40 | 59.12% | 60.92% | +1.79 pp [-8.07, +11.66] | 0.2582 | 1.0000 |
| qwen | joint | 53.71% | 54.50% | +0.79 pp [-18.46, +20.04] | 0.4379 | 1.0000 |
| smol | clean | 83.13% | 83.50% | +0.37 pp [-0.56, +1.31] | -- | -- |
| smol | conflict40 | 65.83% | 68.46% | +2.63 pp [-25.68, +30.93] | 0.3642 | 1.0000 |
| smol | joint | 48.00% | 47.37% | -0.63 pp [-29.51, +28.26] | 0.5328 | 1.0000 |
| tiny | clean | 87.04% | 86.83% | -0.21 pp [-3.61, +3.20] | -- | -- |
| tiny | conflict40 | 59.25% | 68.54% | +9.29 pp [-34.63, +53.22] | 0.2294 | 1.0000 |
| tiny | joint | 54.92% | 37.25% | -17.67 pp [-76.38, +41.05] | 0.8376 | 1.0000 |

Per-seed gains:

- qwen clean: +2.00, +0.13, -1.37 pp.
- qwen conflict40: +6.38, -0.37, -0.62 pp.
- qwen joint: -7.00, +0.87, +8.50 pp.
- smol clean: +0.00, +0.75, +0.37 pp.
- smol conflict40: -8.38, +14.38, +1.88 pp.
- smol joint: -5.38, -9.13, +12.62 pp.
- tiny clean: -1.75, +0.87, +0.25 pp.
- tiny conflict40: +14.38, -10.37, +23.88 pp.
- tiny joint: -42.63, -14.75, +4.37 pp.

## Selection and cost audit

- qwen conflict40: mean exchanged clean fraction 72.9%; train-wall ratio 9.32x; serialized-adapter ratio 2.00x.
- qwen joint: mean exchanged clean fraction 43.6%; train-wall ratio 9.30x; serialized-adapter ratio 2.00x.
- smol conflict40: mean exchanged clean fraction 74.1%; train-wall ratio 4.55x; serialized-adapter ratio 2.00x.
- smol joint: mean exchanged clean fraction 55.7%; train-wall ratio 4.30x; serialized-adapter ratio 2.00x.
- tiny conflict40: mean exchanged clean fraction 75.0%; train-wall ratio 2.84x; serialized-adapter ratio 2.00x.
- tiny joint: mean exchanged clean fraction 44.7%; train-wall ratio 2.90x; serialized-adapter ratio 2.00x.

## Interpretation

Co-Teaching fails either clean safety or a contaminated mean comparison, so it does not provide a uniformly stronger compatible baseline under this protocol.

This baseline uses two complete backbones/adapters and oracle contamination rates. The joint off-task condition is an intentional stress test beyond Co-Teaching's original finite-label-noise setting. Delora is not represented: its published pipeline additionally requires GPT-4o pseudo-label construction and relabeling.

## Source artifacts

- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_clean_seed75.json` (SHA-256 `e052124a9faecdcfbc366419ef879a5737edb081138789de09b871cc8e4d0172`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_clean_seed76.json` (SHA-256 `c150dcea92879349b3f94a36103e2720155978f16b037b0eabb6c5892463eb9b`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_clean_seed77.json` (SHA-256 `f681df49c19dda0c71212f472640f405748fb26433184365c60ac279b07a7e52`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_conflict40_seed75.json` (SHA-256 `c9c545f2ca4c0e35cf2dca700e251506557ef6388057b1fe892f6ccb527abd39`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_conflict40_seed76.json` (SHA-256 `c8d0910cc0ad1b591ec7411098cee2e2a2fc1a191d82792cdbd6e59675b7734d`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_conflict40_seed77.json` (SHA-256 `1f09954f01d2b8c397128247acc8d9c7d5e3d37957066a62b2ca764118f65d88`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_joint_seed75.json` (SHA-256 `38f61bd7fa6496a30074e9fd576b15fd9209cc1f965cefe3faf4e485fcef1942`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_joint_seed76.json` (SHA-256 `e691400276c5137fe806846549139918e009bf05b8245838a6f549019f6df5a4`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_coteach_joint_seed77.json` (SHA-256 `287b709c1d0ff75a9b24653e6c3e9ad336d59be8ea311f700f9b117f16178c61`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_clean_seed75.json` (SHA-256 `82f74891eb47e9814b1bf9d532de63b5004033515e196775170f54365d43f2ed`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_clean_seed76.json` (SHA-256 `affb4abc7e8d2cc628ec0ced4ec2d49db0eaabd699f0828f956443a1ad52a209`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_clean_seed77.json` (SHA-256 `861e8d536697d5d597e740574b03f01f141fabc85513de83198ca2fb22ab0588`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_conflict40_seed75.json` (SHA-256 `643bb30919295fe43882a0a225f53e206dec535b32c5a86efa43d7fff7027240`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_conflict40_seed76.json` (SHA-256 `a17b1f4991682deba622f0593fb7ea0d96ac4ad52b88938e1cab82e2107151ae`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_conflict40_seed77.json` (SHA-256 `51d14fccd5ae89d66a52b768e133f55e3be1cf23160a2cbdd9ab2d7a53c32ae8`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_joint_seed75.json` (SHA-256 `aa6e7e163b5358c2b19fa951214f570f049b17fa0ebb2c85467f2b4c757068e7`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_joint_seed76.json` (SHA-256 `fa8ec4f464d7ab6f0f66ca38198430563dd4c8f0cf7dbb14959e708cf543e7e6`)
- `kafa_script/analysis_outputs/agnews_coteach_qwen_naive_joint_seed77.json` (SHA-256 `fb4866868df2e3a4234cbeea4d95f05fc0245fbc7226f5049535b579b96c47ad`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_clean_seed75.json` (SHA-256 `a5db2b2ee05718945674663314c4a0c7ebe920c0861ba76fd51f308d99226870`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_clean_seed76.json` (SHA-256 `1c24c94f72528193cc1f99a747435731d2784aeccc3d7cdc42dbdbaf583f13d6`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_clean_seed77.json` (SHA-256 `256471fa0214caa1a33fc45ed060e10d0e239f0bf159301f8dcd9c89fba0e4f3`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_conflict40_seed75.json` (SHA-256 `25d9e13240936e74de408be625febfe6baf6ba81d06c9f63f8254a4a20ced103`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_conflict40_seed76.json` (SHA-256 `519a53a9a3dd8ad8c5f122f565026b099ac0af4cf91afa8a29834412da779de5`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_conflict40_seed77.json` (SHA-256 `63c8b7441d3169eef63a1fcadc965995092a29ae3eba4b7b6982932e88e860c8`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_joint_seed75.json` (SHA-256 `d34317de64136b944d9f4d6ae20f42d1ac66dac35271e74a0d795bfa82023aad`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_joint_seed76.json` (SHA-256 `0de5c6bcf3154ea011cefe164399ac2ce8044a348dce9a393f162913111d9814`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_coteach_joint_seed77.json` (SHA-256 `a2043b50d92b8d2ff1e28b6487ee5339dbb8b53adb0bcb80cbff6c2013d6adbf`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_clean_seed75.json` (SHA-256 `30eb2fb69ff91569d71efb9628126cb8a16cfc005e19d2b9d6255a77d4ea6c21`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_clean_seed76.json` (SHA-256 `3b8319473386d3d4e295d1945445f0e5e2e7d9fff9b7e3513c8a91f2a6f2b630`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_clean_seed77.json` (SHA-256 `ad76ed9c87cdb4c8f21f3863c3246ae555d938f2e719e3843637cd03d9c13885`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_conflict40_seed75.json` (SHA-256 `d74093b8df4ffe779e6e988e0a42d7fb6b8b3228e5358bb7688ea9039befcfc6`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_conflict40_seed76.json` (SHA-256 `dc381c9385eb4538170733d3bf8734ecdae7c4e45ef012ce1085e171d65d6c46`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_conflict40_seed77.json` (SHA-256 `ba4b2d9616f2761d1fbec5e1b14e02d4bd97c48b89bc8eaee09710940944a02c`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_joint_seed75.json` (SHA-256 `9ac0fc8ae76681439d4bde384803c822ffc8f36af07b2165b2a490a6a860a2ba`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_joint_seed76.json` (SHA-256 `efe89ebac6753e600935310362bbe711b4234d9d43edcb75a1ae496b23e466e6`)
- `kafa_script/analysis_outputs/agnews_coteach_smol_naive_joint_seed77.json` (SHA-256 `a0e30fb3447f9941c87406fc88452d7ddbb542f40f1eb7daa9c9f81efe340c89`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_clean_seed75.json` (SHA-256 `c5503d60faed78c86c62d71d165eed22b8266056a4907ed474da09788614a9ee`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_clean_seed76.json` (SHA-256 `d818be9cf2c28369aef179b06d099b81f2ab7e15deb158f9b3cc7475047f391c`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_clean_seed77.json` (SHA-256 `d05901a52e8468f7d546ccd930ffc2f2d4c5f1cdfc79715b33408ae368eaefce`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_conflict40_seed75.json` (SHA-256 `ebfe97cd2cfafa63e21afcc37c283a0d7d354b1b595d33bb1044535ef064b042`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_conflict40_seed76.json` (SHA-256 `f38aa6cf841c28942662b6839652ae4b963bb07554fb4b4d129521153bd03b89`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_conflict40_seed77.json` (SHA-256 `deaa1e5bc438099d1edf72a34a252388114654fe0a212309801f0bc6f61c9c9f`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_joint_seed75.json` (SHA-256 `de10eb340519ca070591dfeea95f6624e50f8e0c3e535f448682ceffbd02df18`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_joint_seed76.json` (SHA-256 `e90daa0f5129843e2c65c07e44ea010d9ca54ea74f78eec470da0deb458360d5`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_coteach_joint_seed77.json` (SHA-256 `c78aed7d9039469193a86a414c2414278c8a804d723698c89fbcbe740fbbe9f5`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_clean_seed75.json` (SHA-256 `344e7b1b660f0f94283b242dc8086f00b9edfc34ddf7fd812a8aa6b98af7489d`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_clean_seed76.json` (SHA-256 `b7052b896a3b43de06832d0c4af40ccb3c60876c4457dc93e5179637ff4fca49`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_clean_seed77.json` (SHA-256 `79edbede4e2bae38d3a4d75fa83994938dda6e363dcb5727bfcd5b04aae4ef19`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_conflict40_seed75.json` (SHA-256 `385966d3a6dfcafada86cf79337482b5d0bc46760abaca49061314211a5627c8`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_conflict40_seed76.json` (SHA-256 `a01db4b652a6b11bfdc3e04d53cac73585011894236b929f29bc23d6129b569a`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_conflict40_seed77.json` (SHA-256 `c9891c8782b8d771886e7271b9081bbe684db7668ba12a1c55cbf31032730200`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_joint_seed75.json` (SHA-256 `b9e177bd91f053de66ee5b5d23d9c3963342fbfc82640a1b45e01a69b7320b42`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_joint_seed76.json` (SHA-256 `f09b0c39289ae0ecf98d98371f26fdfa921c2d7cb3e7c9357205f0ea8b0090b2`)
- `kafa_script/analysis_outputs/agnews_coteach_tiny_naive_joint_seed77.json` (SHA-256 `a23a70dab81bf1ecb5fe4fe93ffdbe8a04c10935750a2d72841d8c9c9ddebeec`)
