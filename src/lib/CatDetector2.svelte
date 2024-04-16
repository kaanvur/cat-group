<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import * as mobilenet from '@tensorflow-models/mobilenet';
	import type { MobileNet } from '@tensorflow-models/mobilenet';
	import type { Tensor, Tensor3D } from '@tensorflow/tfjs';

	let mobilenetModel: MobileNet | null = null;
	let image1: Tensor3D | null = null;
	let image2: Tensor3D | null = null;
	let similarityScore: number = 0.0;
	const SIMILARITY_THRESHOLD: number = 0.7;
	let imageUrl1: string = '';
	let imageUrl2: string = '';

	const loadModel = async (): Promise<void> => {
		mobilenetModel = await mobilenet.load();
	};

	onMount(() => {
		loadModel();
	});

	const predictSimilarity = async (): Promise<void> => {
		if (!mobilenetModel || !image1 || !image2) return;

		const resizedImage1: Tensor3D = tf.image.resizeBilinear(image1, [224, 224]);
		const resizedImage2: Tensor3D = tf.image.resizeBilinear(image2, [224, 224]);

		const embeddings1: Tensor = tf.tidy(() => mobilenetModel!.infer(resizedImage1));
		const embeddings2: Tensor = tf.tidy(() => mobilenetModel!.infer(resizedImage2));

		const similarity: Tensor = tf.tidy(() => {
			const dotProduct: Tensor = tf.sum(tf.mul(embeddings1, embeddings2), 1);
			const norm1: Tensor = tf.norm(embeddings1);
			const norm2: Tensor = tf.norm(embeddings2);
			return dotProduct.div(tf.mul(norm1, norm2));
		});

		similarityScore = similarity.dataSync()[0];

		resizedImage1.dispose();
		resizedImage2.dispose();
		embeddings1.dispose();
		embeddings2.dispose();
		similarity.dispose();
	};

	const handleFileChange1 = (event: Event): void => {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files[0]) {
			imageUrl1 = URL.createObjectURL(target.files[0]);
			loadImage(target.files[0], (img: HTMLImageElement) => {
				image1 = tf.browser.fromPixels(img).toFloat().expandDims() as Tensor3D;
				if (image2) predictSimilarity();
			});
		}
	};

	const handleFileChange2 = (event: Event): void => {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files[0]) {
			imageUrl2 = URL.createObjectURL(target.files[0]);
			loadImage(target.files[0], (img: HTMLImageElement) => {
				image2 = tf.browser.fromPixels(img).toFloat().expandDims() as Tensor3D;
				if (image1) predictSimilarity();
			});
		}
	};

	const loadImage = (file: File, callback: (img: HTMLImageElement) => void): void => {
		const reader = new FileReader();
		reader.onload = (e: ProgressEvent<FileReader>) => {
			const img = new Image();
			img.src = e.target!.result as string;
			img.onload = () => callback(img);
		};
		reader.readAsDataURL(file);
	};
</script>

<label for="imageUpload1">Upload first cat image:</label>
<input type="file" id="imageUpload1" accept="image/*" on:change={handleFileChange1} />

<label for="imageUpload2">Upload second cat image:</label>
<input type="file" id="imageUpload2" accept="image/*" on:change={handleFileChange2} />

<div>
	{#if imageUrl1}
		<img src={imageUrl1} alt="First Cat" />
	{/if}
	{#if imageUrl2}
		<img src={imageUrl2} alt="Second Cat" />
	{/if}
</div>

{#if similarityScore > 0}
	<p>Similarity Score: {similarityScore.toFixed(2)}</p>
	{#if similarityScore > SIMILARITY_THRESHOLD}
		<p>This cat looks similar to the previously loaded cat.</p>
	{:else}
		<p>This cat seems different from the previously loaded cat.</p>
	{/if}
{/if}

<style>
	img {
		max-width: 200px;
		height: auto;
	}
</style>
