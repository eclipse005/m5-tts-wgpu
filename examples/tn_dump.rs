//! Debug helper: print the tag / reordered / verbalized stages for one string.
//! usage: cargo run --example tn_dump -- "<text>" [zh|en|ja|de|fr|ko]
use m5_tts_wgpu::tn::Tn;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let text = args.next().unwrap_or_else(|| "1,234 people signed 1/3 of the forms.".into());
    let lang = args.next().unwrap_or_else(|| "English".into());
    let tn = Tn::bundled();
    println!("{}", tn.normalize(&text, &lang, None)?);
    if std::env::var("TN_STAGES").is_ok() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tn_data");
        let prefix = match lang.as_str() { "English"|"en" => "en_tn", "Chinese"|"zh" => "zh_tn", "Japanese"|"ja" => "ja_tn", "German"|"de" => "de_tn", "French"|"fr" => "fr_tn", _ => "ko_tn" };
        let n = m5_tts_wgpu::tn::Normalizer::load(&dir, prefix)?;
        let tag = n.tag(&text)?;
        println!("TAG      : {tag}");
        println!("REORDERED: {}", n.reorder_tagged(&tag));
        println!("VERBAL   : {}", n.verbalize(&tag)?);
    }
    Ok(())
}
