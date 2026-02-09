import runpod
import traceback

net = None
feature_utils = None
seq_cfg = None
INIT_ERROR = None


def load_model():
    global net, feature_utils, seq_cfg

    print("[init] Importing torch...", flush=True)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda'
    dtype = torch.bfloat16

    print("[init] Importing mmaudio...", flush=True)
    from mmaudio.eval_utils import ModelConfig, all_model_cfg, setup_eval_logging
    from mmaudio.model.networks import MMAudio, get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    print("[init] Loading model config...", flush=True)
    model: ModelConfig = all_model_cfg['large_44k_v2']
    model.download_if_needed()
    setup_eval_logging()

    print("[init] Loading network weights...", flush=True)
    seq_cfg = model.seq_cfg
    net = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))

    print("[init] Loading feature utils (CLIP, VAE, etc.)...", flush=True)
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device, dtype).eval()

    print("[init] Model ready.", flush=True)


try:
    load_model()
except Exception:
    INIT_ERROR = traceback.format_exc()
    print(f"[init] FAILED:\n{INIT_ERROR}", flush=True)


def handler(job):
    if INIT_ERROR:
        return {'error': f'Model failed to load:\n{INIT_ERROR}'}

    import base64
    import io
    import torch
    import torchaudio
    from mmaudio.eval_utils import generate
    from mmaudio.model.flow_matching import FlowMatching

    job_input = job['input']

    prompt = job_input.get('prompt', '')
    if not prompt:
        return {'error': 'prompt is required'}

    negative_prompt = job_input.get('negative_prompt', '')
    duration = float(job_input.get('duration', 8.0))
    num_steps = int(job_input.get('num_steps', 25))
    cfg_strength = float(job_input.get('cfg_strength', 4.5))
    seed = int(job_input.get('seed', -1))

    try:
        with torch.inference_mode():
            rng = torch.Generator(device='cuda')
            if seed >= 0:
                rng.manual_seed(seed)
            else:
                rng.seed()

            fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

            seq_cfg.duration = duration
            net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

            audios = generate(
                None, None, [prompt],
                negative_text=[negative_prompt],
                feature_utils=feature_utils,
                net=net, fm=fm, rng=rng,
                cfg_strength=cfg_strength,
            )
            audio = audios.float().cpu()[0]

            buf = io.BytesIO()
            torchaudio.save(buf, audio, seq_cfg.sampling_rate, format='wav')
            buf.seek(0)
            wav_bytes = buf.read()

        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
    except Exception as e:
        return {'error': str(e)}

    return {
        'audio_base64': audio_base64,
        'sample_rate': seq_cfg.sampling_rate,
        'duration': duration,
        'format': 'wav',
    }


runpod.serverless.start({'handler': handler})
