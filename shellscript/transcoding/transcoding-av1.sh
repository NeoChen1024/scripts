#!/bin/sh
# "HD-VCD": squeeze a video into a VCD-ish bitrate with two-pass VBR AV1 (SVT-AV1 or NVENC)

usage() {
	echo "Usage: $0 [-e encoder] [-b bitrate] [-a audio_bitrate] [-p preset] [-f film_grain] [-d] [-s height] [-r fps] [-o output] input" >&2
	echo "  -e  video encoder: svt (SVT-AV1) or nvenc (NVENC AV1 with NVDEC decoding) (default: svt)" >&2
	echo "  -b  video target bitrate (default: 256k)" >&2
	echo "  -a  Opus audio bitrate (default: 64k)" >&2
	echo "  -p  encoder preset (default: 4 for SVT-AV1, p7 for NVENC)" >&2
	echo "  -f  film grain synthesis level, 0 = off (default: 0, typical: 8-12; SVT-AV1 only)" >&2
	echo "  -d  denoise source before film grain estimation (SVT-AV1 only)" >&2
	echo "  -s  downscale to this height (default: keep source resolution)" >&2
	echo "  -r  convert to this frame rate, e.g. 30, 30000/1001, ntsc (default: keep source frame rate)" >&2
	echo "  -o  output file (default: <input without extension>-small.mkv)" >&2
	exit 1
}

encoder=svt
bitrate=256k
audio_bitrate=64k
preset=
film_grain=0
film_grain_denoise=0
height=
fps=
output=

while getopts "e:b:a:p:f:ds:r:o:h" opt; do
	case "$opt" in
		e) encoder="$OPTARG" ;;
		b) bitrate="$OPTARG" ;;
		a) audio_bitrate="$OPTARG" ;;
		p) preset="$OPTARG" ;;
		f) film_grain="$OPTARG" ;;
		d) film_grain_denoise=1 ;;
		s) height="$OPTARG" ;;
		r) fps="$OPTARG" ;;
		o) output="$OPTARG" ;;
		*) usage ;;
	esac
done
shift $((OPTIND - 1))
[ -n "$1" ] || usage
input="$1"
[ -n "$output" ] || output="${input%.*}-small.mkv"

case "$encoder" in
	svt) : "${preset:=4}" ;;
	nvenc)
		: "${preset:=p7}"
		[ "$film_grain" = 0 ] && [ "$film_grain_denoise" = 0 ] ||
			echo "$0: film grain synthesis isn't supported by NVENC, ignoring -f/-d" >&2
		;;
	*) usage ;;
esac

params="tune=0:keyint=10s:scd=1:enable-qm=1:enable-variance-boost=1:variance-octile=6:luminance-qp-bias=20:ac-bias=0.5"
params="$params:film-grain=$film_grain:film-grain-denoise=$film_grain_denoise"

vf="null"
[ -n "$fps" ] && vf="$vf,fps=$fps"
[ -n "$height" ] && vf="$vf,scale=-2:'min($height,ih)':flags=lanczos"

passlog="$(mktemp -u "${TMPDIR:-/tmp}/av1-2pass.XXXXXX")"
trap 'rm -f "$passlog"-*.log' EXIT

mux() {
	ffmpeg \
		${hwaccel:+-hwaccel "$hwaccel"} \
		-i "$input" \
		-c:a libopus -b:a "$audio_bitrate" -ac 2 \
		-vf "$vf" \
		"$@" \
		-c:s srt \
		-cues_to_front 1 \
		"$output"
}

if [ "$encoder" = nvenc ]; then
	# NVDEC decodes when it supports the source and FFmpeg falls back to software otherwise;
	# filters stay on the CPU so both paths share one chain. NVENC does its own two-pass
	# (multipass fullres); a huge GOP plus forced IDRs gives a keyframe every 10 seconds.
	hwaccel=cuda
	vf="$vf,format=p010le"
	mux \
		-c:v av1_nvenc -preset "$preset" -tune uhq -rc vbr -b:v "$bitrate" \
		-multipass fullres -rc-lookahead 32 -spatial-aq 1 -temporal-aq 1 \
		-g 1000000 -forced-idr 1 -force_key_frames 'expr:gte(t,n_forced*10)'
	exit
fi

svt() {
	mux -pix_fmt yuv420p10le -c:v libsvtav1 -preset "$preset" -b:v "$bitrate" -svtav1-params "$params" "$@"
}

# libsvtav1 two-pass support landed in FFmpeg 8.1; older releases silently ignore -pass,
# which would turn both passes into full one-pass encodes. Git snapshots ("N-...") are assumed new enough.
ffmpeg_version="$(ffmpeg -version | sed -n '1s/^ffmpeg version n\{0,1\}\([^ ]*\).*/\1/p')"
case "$ffmpeg_version" in
	[0-9]*)
		major="${ffmpeg_version%%.*}"
		minor="${ffmpeg_version#*.}"; minor="${minor%%[!0-9]*}"
		[ "$major" -gt 8 ] || { [ "$major" -eq 8 ] && [ "${minor:-0}" -ge 1 ]; } && two_pass=1 ;;
	*) two_pass=1 ;;
esac

if [ -n "$two_pass" ]; then
	ffmpeg \
		-i "$input" \
		-an -sn \
		-vf "$vf" -pix_fmt yuv420p10le \
		-c:v libsvtav1 -preset "$preset" -b:v "$bitrate" -svtav1-params "$params" \
		-pass 1 -passlogfile "$passlog" \
		-f null - &&
	svt -pass 2 -passlogfile "$passlog"
else
	echo "$0: FFmpeg $ffmpeg_version lacks libsvtav1 two-pass support (needs 8.1+), falling back to one-pass VBR" >&2
	svt
fi
