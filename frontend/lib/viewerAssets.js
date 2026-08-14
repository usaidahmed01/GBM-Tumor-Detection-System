import { proxyApiUrl } from './api';

export const MRI_SEQUENCE_OPTIONS = [
  { value: 'T1C', alias: 'mri_t1c', label: 'T1C', description: 'Post-contrast' },
  { value: 'T1', alias: 'mri_t1', label: 'T1', description: 'Anatomy' },
  { value: 'T2', alias: 'mri_t2', label: 'T2', description: 'Fluid-sensitive' },
  { value: 'FLAIR', alias: 'mri_flair', label: 'FLAIR', description: 'Edema-sensitive' },
];

export const SEGMENT_LEGEND = [
  { index: 2, region: 'WT', label: 'Whole Tumor', color: '#27d3d1' },
  { index: 1, region: 'TC', label: 'Tumor Core', color: '#ffbd6a' },
  { index: 4, region: 'ET', label: 'Enhancing Tumor', color: '#ff5cab' },
];

export function assetByAlias(manifest, alias) {
  return manifest?.assets?.find((asset) => asset.alias === alias) || null;
}

export function loaderUrlForAsset(asset) {
  return proxyApiUrl(asset?.loader_url || asset?.download_url);
}

export function mriAssetForSequence(manifest, sequence) {
  const option = MRI_SEQUENCE_OPTIONS.find((item) => item.value === sequence);
  return option ? assetByAlias(manifest, option.alias) : null;
}
