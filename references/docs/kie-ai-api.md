# Kie AI API Reference

## Common
- **Auth**: `Authorization: Bearer YOUR_API_KEY`
- **Create Task**: `POST https://api.kie.ai/api/v1/jobs/createTask`
- **Query Status**: `GET https://api.kie.ai/api/v1/jobs/recordInfo?taskId=TASK_ID`
- **States**: `waiting` -> `success` | `fail`
- **File Upload**: `POST https://kieai.redpandaai.co/api/file-stream-upload` (requires `uploadPath` in form data)

## Nano Banana Pro (Image Generation)
- **Model**: `nano-banana-pro`
- **Input**: prompt (required, max 20000 chars), image_input (array of URLs, up to 8), aspect_ratio, resolution (1K/2K/4K), output_format (png/jpg)
- **Cost**: $0.09/image

## Kling 3.0 (Image/Text-to-Video)
- **Model**: `kling-3.0/video`
- **Input**:
  - mode: "std" or "pro" (required)
  - prompt: text prompt (required for single shot, max 2500 chars)
  - image_urls: array of image URLs for start/end frames
  - duration: "3"-"15" seconds
  - multi_shots: boolean (false = single shot, true = multi-shot)
  - multi_prompt: array of {prompt, duration} objects (for multi-shot mode)
  - sound: boolean
  - aspect_ratio: "16:9", "9:16", "1:1" (ignored when image_urls provided)
  - kling_elements: array of {name, description, element_input_urls/element_input_video_urls} for @element_name references in prompts

## Sora 2 Pro (Image-to-Video)
- **Model**: `sora-2-pro-image-to-video`
- **Input**:
  - prompt: text prompt (required, max 10000 chars)
  - image_urls: array of image URLs for first frame (required)
  - aspect_ratio: "portrait" or "landscape"
  - n_frames: "10" (10s) or "15" (15s)
  - size: "standard" or "high"
  - remove_watermark: boolean (default true)
  - upload_method: "s3" or "oss"

## Sora 2 Pro (Text-to-Video) - alternate endpoint
- **Model**: `sora-2-pro-text-to-video`
- **Input**: Same as image-to-video but without image_urls
