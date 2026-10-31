//! Optional private-repository authentication. Credentials are never app data.
use anyhow::Result;

/// Public downloads work anonymously. For an owner's private model repository,
/// reuse an existing OS credential without storing a token in the application.
#[cfg(windows)]
pub fn read_huggingface_token() -> Result<Option<String>> {
    use anyhow::bail;
    use windows_sys::Win32::{
        Foundation::{ERROR_NOT_FOUND, GetLastError},
        Security::Credentials::{CRED_TYPE_GENERIC, CREDENTIALW, CredFree, CredReadW},
    };
    for target in ["BRP.CHROMA/HuggingFace", "BRP.Canvas/HuggingFace"] {
        let name: Vec<u16> = target.encode_utf16().chain(Some(0)).collect();
        let mut pointer: *mut CREDENTIALW = std::ptr::null_mut();
        // Windows allocates the record; its credential blob remains valid until
        // CredFree. Only these application-specific targets are ever queried.
        let found = unsafe { CredReadW(name.as_ptr(), CRED_TYPE_GENERIC, 0, &mut pointer) };
        if found == 0 {
            let error = unsafe { GetLastError() };
            if error == ERROR_NOT_FOUND {
                continue;
            }
            bail!("The saved Hugging Face sign-in could not be read (Windows error {error}).");
        }
        let mut bytes = unsafe {
            let record = &*pointer;
            if record.CredentialBlobSize == 0 || record.CredentialBlob.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(
                    record.CredentialBlob,
                    record.CredentialBlobSize as usize,
                )
                .to_vec()
            }
        };
        unsafe { CredFree(pointer.cast()) };
        let result = decode_token(&bytes);
        bytes.fill(0);
        return result.map(Some);
    }
    Ok(None)
}

#[cfg(not(windows))]
pub fn read_huggingface_token() -> Result<Option<String>> {
    Ok(None)
}

#[cfg(any(windows, test))]
fn decode_token(bytes: &[u8]) -> Result<String> {
    use anyhow::{Context, ensure};
    let decoded = if bytes.contains(&0) {
        ensure!(
            bytes.len().is_multiple_of(2),
            "Invalid saved Hugging Face sign-in."
        );
        let units: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        String::from_utf16(&units).context("Invalid saved Hugging Face sign-in.")?
    } else {
        String::from_utf8(bytes.to_vec()).context("Invalid saved Hugging Face sign-in.")?
    };
    let token = decoded.trim_matches('\0').trim();
    ensure!(
        token.starts_with("hf_")
            && token.len() > 6
            && token[3..].bytes().all(|byte| byte.is_ascii_alphanumeric()),
        "The saved Hugging Face sign-in has an unsupported format."
    );
    Ok(token.to_string())
}

#[cfg(test)]
mod tests {
    use super::decode_token;
    #[test]
    fn existing_windows_credentials_accept_utf8_and_utf16_without_serialization() {
        let fixture = "hf_TestFixtureOnly123";
        assert_eq!(decode_token(fixture.as_bytes()).unwrap(), fixture);
        let wide: Vec<u8> = fixture
            .encode_utf16()
            .chain(Some(0))
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(decode_token(&wide).unwrap(), fixture);
    }
    #[test]
    fn malformed_credentials_do_not_echo_their_content_in_errors() {
        let rejected = "invalid-sensitive-fixture";
        let error = decode_token(rejected.as_bytes()).unwrap_err().to_string();
        assert!(!error.contains(rejected));
        assert!(decode_token(&[1, 0, 2]).is_err());
        assert!(decode_token(b"hf_bad\nheader").is_err());
    }
}
