use crate::Message;
use crate::image_reader::Frame;
use iced::widget::{Space, column, container, image, row, text};
use iced::{Alignment, Element, Length};

pub fn images_grid(frames: &'_ [Frame], column_count: usize) -> Element<'_, Message> {
    column(frames.chunks_exact(column_count).map(|chunk| {
        let mut r = row(chunk.iter().map(|frame| {
            container(column![
                text(&frame.name)
                    .width(Length::Fill)
                    .align_x(Alignment::Center),
                image(frame.image.clone())
                    .width(Length::Fill)
                    .content_fit(iced::ContentFit::Contain),
            ])
            .align_x(Alignment::Center)
            .padding(0.5f32)
            .into()
        }));

        for _ in 0..column_count - chunk.len() {
            r = r.push(Space::with_width(Length::Fill));
        }

        r.into()
    }))
    .into()
}
