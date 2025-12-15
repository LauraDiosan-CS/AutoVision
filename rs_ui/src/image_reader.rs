use crate::decoder::{bgr_to_rgba, grayscale_to_rgba};
use crate::Message;
use iced::futures;
use iced::futures::channel::mpsc;
use iced::widget::image;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Py, PyRef, Python};
use std::collections::BTreeMap;
use std::thread;

#[derive(Debug, Clone, Default)]
pub struct Frames {
    pub description: String,
    pub frames: Vec<Frame>,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub name: String,
    pub image: image::Handle,
}

#[pyclass]
struct FrameReceiver {
    sender: mpsc::UnboundedSender<Message>,
    cache: BTreeMap<String, image::Handle>,
}

#[pyclass]
pub struct PyFrame {
    pub name: String,
    pub bytes: Py<PyBytes>, // Keeps reference to Python bytes without copying
    pub width: u32,
    pub height: u32,
    pub channels: u8,
}

#[pymethods]
impl PyFrame {
    #[new]
    fn new(name: String, bytes: Py<PyBytes>, width: u32, height: u32, channels: u8) -> Self {
        PyFrame {
            name,
            bytes,
            width,
            height,
            channels,
        }
    }
}

#[pymethods]
impl FrameReceiver {
    fn send_frames(
        &mut self,
        py: Python<'_>,
        description: String,
        raw_frames: Vec<PyRef<PyFrame>>,
    ) -> bool {
        if self.sender.is_closed() {
            return false;
        }

        for py_frame in raw_frames {
            let pixel_data = py_frame.bytes.as_bytes(py);

            let expected_size = py_frame.width as usize * py_frame.height as usize * py_frame.channels as usize;
            if pixel_data.len() != expected_size {
                eprintln!("Invalid image size: {} instead of {} ({})", pixel_data.len(), expected_size, py_frame.name);
                continue;
            }

            let handle = image::Handle::from_rgba(
                py_frame.width,
                py_frame.height,
                match py_frame.channels {
                    1 => grayscale_to_rgba(pixel_data),
                    3 => bgr_to_rgba(pixel_data),
                    4 => pixel_data.to_vec().into_boxed_slice(),
                    _ => {
                        eprintln!(
                            "Invalid image ({}) with {} channels",
                            py_frame.name, py_frame.channels
                        );
                        continue;
                    }
                },
            );

            self.cache.insert(py_frame.name.clone(), handle);
        }

        let main_frame_name = "Main";
        let mut frames: Vec<_> = self
            .cache
            .iter()
            .map(|(name, image)| Frame {
                name: name.clone(),
                image: image.clone(),
            })
            .collect();
        if let Some(index) = frames
            .iter()
            .position(|frame| frame.name == main_frame_name)
        {
            let element = frames.remove(index);
            frames.insert(0, element);
        }
        let frames = Frames {
            description,
            frames,
        };

        if self.sender.unbounded_send(Message::NewFrames(frames)).is_err() {
            return false;
        }

        true
    }

    fn stop(&mut self) {
        let _ = self.sender.unbounded_send(Message::VideoFinished);
    }

    fn is_stopped(&self) -> bool {
        self.sender.is_closed()
    }
}

pub fn image_producer() -> impl futures::Stream<Item = Message> {
    let (sender, receiver) = mpsc::unbounded();

    thread::spawn(move || {
        Python::attach(|py| {
            let sys = py.import("sys").expect("Failed to import sys");
            let paths = sys.getattr("path").expect("Failed to get path");
            if let Err(e) = paths.call_method1("append", (".",)) {
                eprintln!("Failed to append path: {}", e);
                return;
            }

            let main_module = py.import("main_rust").expect("Failed to import");

            let callback = FrameReceiver {
                sender,
                cache: BTreeMap::new(),
            };
            let py_callback = Py::new(py, callback).expect("Failed to create Py callback");

            let py_frame_cls = py.get_type::<PyFrame>();

            main_module
                .call_method1("run_from_rust", (py_callback, py_frame_cls))
                .expect("Failed to call run_from_rust");
        });
    });

    receiver
}
