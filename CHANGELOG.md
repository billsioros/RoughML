# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--next-version-placeholder-->

## v4.1.0 (2021-07-27)
### Feature
* Implement the tuning flow ([`81bf9c0`](https://github.com/billsioros/thesis/commit/81bf9c08bec11d414537e99555bced31be857e0b))
* Create a hyper parameter tuning CLI ([`5f462d1`](https://github.com/billsioros/thesis/commit/5f462d19e88ec57560a90f46115ba1889bf5bea4))
* Dataset generation script ([`428b6fd`](https://github.com/billsioros/thesis/commit/428b6fd45673d80f60ee81c6a293f34ae8b32f03))
* `cli`s aimed towards visualization and dataset generation ([`692a1c0`](https://github.com/billsioros/thesis/commit/692a1c00d27680276fa81bb85c9898c4c7741aa2))

### Fix
* Make tuning CLI available after installation ([`1b90a34`](https://github.com/billsioros/thesis/commit/1b90a34165398518db46f171bbfdcb293db64e6e))
* Show default logging level ([`986b6ac`](https://github.com/billsioros/thesis/commit/986b6ac23c96487311821211f65e867aae1d4817))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v4.0.0...v4.1.0)**

## v4.0.0 (2021-07-18)
### Feature
* Add hyper parameter tuning interface ([`d486372`](https://github.com/billsioros/thesis/commit/d486372e55c1d3a6e0ba3e233d8b1c266fe16579))
* Make content loss optional ([`75d9b0e`](https://github.com/billsioros/thesis/commit/75d9b0e9bc843f1a1605d2fbdbd7825d1866de0d))

### Breaking
* make optimizer a parameter to `TrainingManager` ([`57f0dc8`](https://github.com/billsioros/thesis/commit/57f0dc80aeb33a9328147043b1e2d40bd429df3f))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v3.1.0...v4.0.0)**

## v3.1.0 (2021-07-15)
### Feature
* Separatelly log/plot `content_loss` ([`fc16be6`](https://github.com/billsioros/thesis/commit/fc16be6142c7435d900ef1538295c9826ede3d60))
* Instantiate models per dataset ([`f7bee53`](https://github.com/billsioros/thesis/commit/f7bee5310ed03ea9f3cbee64361d8b26d1ec5798))
* Optionally suppress exceptions on training flow ([`122a055`](https://github.com/billsioros/thesis/commit/122a0556742b85a579576a6cf96b28ac755580d5))
* Separate limits for dataset/surface loading ([`4754d72`](https://github.com/billsioros/thesis/commit/4754d72ec3383192fb5ceb6385ee8a6daeb2f5d3))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v3.0.1...v3.1.0)**

## v3.0.1 (2021-07-10)
### Fix
* Overwrite dataset transforms if provided ([`ed9dc7b`](https://github.com/billsioros/thesis/commit/ed9dc7b37713ae39261cd1f8036a244a8470e12b))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v3.0.0...v3.0.1)**

## v3.0.0 (2021-07-10)
### Feature
* Notify on flow completion (via email) ([`38fe3a2`](https://github.com/billsioros/thesis/commit/38fe3a2847c7e140967c67f2cd339842467dd4ed))

### Breaking
* notify on flow completion (via email) ([`38fe3a2`](https://github.com/billsioros/thesis/commit/38fe3a2847c7e140967c67f2cd339842467dd4ed))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v2.2.0...v3.0.0)**

## v2.2.0 (2021-07-08)
### Feature
* Run the flow over multiple datasets ([`d01651d`](https://github.com/billsioros/thesis/commit/d01651d9fbbb6b33fff1c51ebd48d4edd9f5b123))
* Save training flow outputs to Google Drive by default ([`1032892`](https://github.com/billsioros/thesis/commit/103289203d6ef5dbd90de4c82dce5fce358c8b92))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v2.1.0...v2.2.0)**

## v2.1.0 (2021-07-01)
### Feature
* Save images on flow exit ([`83127d9`](https://github.com/billsioros/thesis/commit/83127d9abde626a80d22617fa5a4461db7153eb1))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v2.0.0...v2.1.0)**

## v2.0.0 (2021-07-01)
### Feature
* Enable gradient clipping ([`14dc13f`](https://github.com/billsioros/thesis/commit/14dc13fe7aabf2d47e30f825bfce3f3515cd4fbc))

### Breaking
* convert `TrainingManager.__call__` to iterator ([`04b18c9`](https://github.com/billsioros/thesis/commit/04b18c90895a0e8a5e06b381d1684aa6fd6c3d98))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.5.2...v2.0.0)**

## v1.5.2 (2021-06-29)
### Fix
* Wrong path when invoking `pip` ([`59cb5e3`](https://github.com/billsioros/thesis/commit/59cb5e347f140ba4e85ae9349ac64557f885ec74))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.5.1...v1.5.2)**

## v1.5.1 (2021-06-29)
### Fix
* Criterion and content loss weights add up to 1 ([`2447088`](https://github.com/billsioros/thesis/commit/244708899463815658d01134842a6a5af60970f6))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.5.0...v1.5.1)**

## v1.5.0 (2021-06-28)
### Feature
* Add title to per epoch animation ([`52b4bc3`](https://github.com/billsioros/thesis/commit/52b4bc35d07b157e38a090d262a6be1074f9642e))
* Cache content loss(es) ([`b5c9fed`](https://github.com/billsioros/thesis/commit/b5c9fed1777f2944fa117d4ab916d919a7b9e85b))

### Fix
* Check instance class when loading `from_pickle` ([`0dd3c8d`](https://github.com/billsioros/thesis/commit/0dd3c8d9a604e9ca7ce18c0b7dacfc749e7b1deb))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.4.0...v1.5.0)**

## v1.4.0 (2021-06-27)
### Feature
* Store/load dataset to/from `.pt` file ([`4b17918`](https://github.com/billsioros/thesis/commit/4b17918c6005dafb632b251c13e4e73f42cd2db8))
* Save per epoch animation(s) ([`dedc063`](https://github.com/billsioros/thesis/commit/dedc063c6434a7852758bc5110a4de061da9842a))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.3.0...v1.4.0)**

## v1.3.0 (2021-06-24)
### Feature
* Configure logging level via `export` ([`312f175`](https://github.com/billsioros/thesis/commit/312f175b368176d6bf42b2bd880b887e2160f444))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.2.0...v1.3.0)**

## v1.2.0 (2021-06-23)
### Feature
* Per epoch surface batch animation ([`448d3dc`](https://github.com/billsioros/thesis/commit/448d3dcbe7cbc90a8e8ad48d66b61d73e8fa84c1))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.1.0...v1.2.0)**

## v1.1.0 (2021-06-20)
### Feature
* Add parallel hpg content loss ([`250e297`](https://github.com/billsioros/thesis/commit/250e2973a99bcf93ffd08e4c4c15205f645f1c5d))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.0.1...v1.1.0)**

## v1.0.1 (2021-06-20)
### Fix
* Removed `src` import(s) prefix ([`cc0c918`](https://github.com/billsioros/thesis/commit/cc0c9186fccc16cfd2b9061d79f44a678f05065c))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v1.0.0...v1.0.1)**

## v1.0.0 (2021-06-20)
### Fix
* Set python minimum version to 3.7 ([`d81db21`](https://github.com/billsioros/thesis/commit/d81db219881adff2e094800df75a335fdffdf2f6))

### Breaking
* set python minimum version to 3.7 ([`d81db21`](https://github.com/billsioros/thesis/commit/d81db219881adff2e094800df75a335fdffdf2f6))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v0.2.0...v1.0.0)**

## v0.2.0 (2021-06-19)
### Feature
* Convert to library ([`7aee276`](https://github.com/billsioros/thesis/commit/7aee2764697e2ca23e6000ca6695c13823a9e3f6))
* Add `picke`-ing ([`3f9d302`](https://github.com/billsioros/thesis/commit/3f9d3029b5edccfe7d7bf30c4685aefd68c308ff))
* Added different flavors of content losses ([`721c529`](https://github.com/billsioros/thesis/commit/721c5295b236998ce539879fdb9ac19b24382946))
* Added CLI ([`3cfd8d9`](https://github.com/billsioros/thesis/commit/3cfd8d99412dc0b3bd7d2738dc979429ad529e84))
* Add `statistics` logging ([`d7d763b`](https://github.com/billsioros/thesis/commit/d7d763b68408f02777922f12893fa13de8b85594))
* Add timing ([`9bfb1d6`](https://github.com/billsioros/thesis/commit/9bfb1d69c91e211f149467f8ff8d1f6563b8be3a))
* Add pretrained HPG2D playground ([`6289c8b`](https://github.com/billsioros/thesis/commit/6289c8b3915c8520a84867a13b8e71c5315b61ff))

### Fix
* Initial release ([`7995a9f`](https://github.com/billsioros/thesis/commit/7995a9f6b5b8a32c43b3b9b4e0ce2a2203af57f1))
* Initial version ([`b3d7a3e`](https://github.com/billsioros/thesis/commit/b3d7a3e254af50147a441bc0b1f1a295a08fd288))
* Sync `py` and `ipynb` versions ([`6092641`](https://github.com/billsioros/thesis/commit/6092641ae734d6267473c3dadbbf5c9603a84543))

**[See all commits in this version](https://github.com/billsioros/thesis/compare/v0.1.0...v0.2.0)**
