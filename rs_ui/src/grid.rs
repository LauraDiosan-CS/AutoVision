use crate::Message;
use iced::widget::{column, container, image, responsive, row, text};
use iced::{Alignment, Element, Length};
use crate::image_reader::Frame;

pub fn images_grid(frames: &'_ [Frame], column_count: usize) -> Element<'_, Message> {
    // responsive(move |size| {
    column(frames.chunks(column_count).map(|chunk| {
        row(chunk.iter().map(|frame| {
            container(column![
                image(frame.image.clone())
                    .width(Length::Fill)
                    .content_fit(iced::ContentFit::Contain),
                text(&frame.name).width(Length::Fill).align_x(Alignment::Center)
            ])
            .padding(1)
            .into()
        }))
        .into()
    }))
    .into()
    // })
    // .into()
}
